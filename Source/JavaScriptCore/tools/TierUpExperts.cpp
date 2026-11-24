/*
 * minuteman: structurally different tier-up and deopt experts with ghost votes.
 * See TierUpExperts.h for the experts, decision points and fixed constants.
 */

#include "config.h"
#include "TierUpExperts.h"

#if ENABLE(DFG_JIT)

#include "ArrayProfile.h"
#include "BaselineJITCode.h"
#include "CodeBlock.h"
#include "CodeBlockInlines.h"
#include "DFGCommonData.h"
#include "DFGExitProfile.h"
#include "DFGJITCode.h"
#include "DFGOSRExitBase.h"
#include "ExecutionCounter.h"
#include "InlineCallFrame.h"
#include "JITWorklist.h"
#include "JSCInlines.h"
#include "LazyOperandValueProfile.h"
#include "LazyValueProfile.h"
#include "MethodOfGettingAValueProfile.h"
#include "PropertyInlineCache.h"
#include "TierUpTimeAttribution.h"
#include <cstdlib>
#include <cstring>
#include <queue>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>

namespace JSC {

// Read-only view of an exit's value-profile locator (friend of MethodOfGettingAValueProfile).
struct MinutemanValueProfileAccess {
    using Kind = MethodOfGettingAValueProfile::Kind;
    static Kind kind(const MethodOfGettingAValueProfile& m) { return m.m_kind; }
    static const CodeOrigin& codeOrigin(const MethodOfGettingAValueProfile& m) { return m.m_codeOrigin; }
    static uint64_t rawOperand(const MethodOfGettingAValueProfile& m) { return m.m_rawOperand; }
};

namespace TierUpExperts {

bool g_enabled { false };

namespace {

// ---------------------------------------------------------------- fixed constants
constexpr unsigned kSupplySlots = 4;          // Caller-Pull: tracked cold inlinable callees per caller
constexpr double kEwmaWeight = 1.0 / 8;       // Queue-Slack: Jacobson's TCP RTT gain

enum TierExpert : unsigned { TJsc, TSupply, TQuiescence, TSampleGravity, TQueueSlack, NumTierExperts };
enum DeoptExpert : unsigned { DJsc, DRefutation, DCauseSplit, NumDeoptExperts };
enum Vote : uint8_t { Abstain = 0, Yes = 1, No = 2 }; // Yes = Go / Jettison, No = Wait / Keep

const char* const kTierNames[] = { "jsc-default", "supply", "quiescence", "sample-gravity", "queue-slack" };
const char* const kDeoptNames[] = { "jsc-default", "refutation", "cause-split" };
const char* const kDPNames[] = { "LLIntToBaseline", "BaselineToDFG", "BaselineLoopOSR", "DFGToFTL", "DFGLoopOSR", "Retier", "Exit", "LoopReoptimize" };

unsigned s_tierSelected { TJsc };
unsigned s_deoptSelected { DJsc };

// ---------------------------------------------------------------- state
// Raw ghost losses for one function, in ms of sampled time, priced at the end of the run with
// this run's measured tier speedups (every loss is extra time). For relative regret, each
// expert's loss is paired with jsc-default's loss at the decision points where that expert was
// awake ("...Jsc[e]").
struct EarlyEvent {
    uint8_t step;
    uint8_t yes;       // experts that voted Go for the jettisoned compile
    uint8_t awake;     // experts awake at that decision point
    double lowerMs[2]; // exit time while installed + time until re-tier, by lower tier (Baseline, DFG)
};

struct RawLoss {
    double late[2][5] { };          // lower-tier ms while waiting after some expert voted Go
    double lateJsc[2][5] { };
    double delayed[2][2][5] { };    // [step][higher tier: DFG, FTL][e] higher-tier ms while still waiting
    double delayedJsc[2][2][5] { };
    Vector<EarlyEvent> early;
    unsigned synthetic[2][5] { };   // superseded un-acted Go votes
    unsigned syntheticJsc[2][5] { };
};

struct FnState { // per baseline CodeBlock
    // supply
    CodeBlock* slots[kSupplySlots] { };
    bool deferred[kSupplySlots] { };
    unsigned nextSlot { 0 };
    bool pulled { false };
    HashSet<CodeBlock*> trackedBy; // callers whose slots hold this function
    // quiescence
    uint64_t fingerprint { 0 };
    bool haveFingerprint { false };
    MonotonicTime lastChange;
    bool requireChange { false };
    // queue-slack
    uint64_t lastJettisonTickPlusOne { 0 };
    // ---- ghost scoring. Step si: 0 = Baseline->DFG, 1 = DFG->FTL. Tier index t: 0 Baseline, 1 DFG, 2 FTL.
    Vote latest[2][5] { };          // each tier-up expert's vote at the most recent decision point
    uint8_t latestAwake[2] { 0, 0 }; // experts awake (not abstaining) at that point
    bool haveLatest[2] { false, false };
    uint8_t compileYesMask[2] { 0, 0 };   // at the acted Go: experts that voted Go
    uint8_t compileAwakeMask[2] { 0, 0 }; // at the acted Go: experts awake
    bool ghostActive[2][5] { };     // voted Go while the acted decision was Wait
    uint64_t ghostFingerprint[2][5] { };
    uint64_t ghostPoint[2][5] { };
    uint8_t waiting[2] { 0, 0 };      // voted Wait at the acted Go and have not re-voted Go since
    uint8_t waitingAwake[2] { 0, 0 }; // experts awake at that acted Go
    // jettison cost accounting: lower-tier time (ms) by lower tier (0 Baseline, 1 DFG)
    bool installed[2] { false, false };
    double installedLowerMs[2][2] { };  // exits finishing in a lower tier while installed
    bool pendingEarly[2] { false, false };
    uint8_t pendingYes[2] { 0, 0 }, pendingAwake[2] { 0, 0 };
    double pendingLowerMs[2][2] { };    // the above, plus lower-tier time since the jettison
    RawLoss raw;
    double sampleMs[3] { };             // this function's sampled ms in Baseline / DFG / FTL
    uint64_t retiredWork[2] { 0, 0 };   // work of destroyed DFG / FTL code for this function
    HashSet<CodeBlock*> liveOptimized;  // optimized CodeBlocks with a work counter
    int8_t functionClass { -1 };
    Vector<CodeBlock*> inlinees;    // inlined by this function's most recent optimized compile
    HashSet<CodeBlock*> inlinedInto; // reverse of `inlinees`
    // cause-split retier
    bool retierPending { false };
    bool retierHasValue { false };
    SpeculatedType retierValue { SpecNone };
    CodeBlock* retierProfiled { nullptr };
    BytecodeIndex retierIndex;
};

struct OptState { // per optimized CodeBlock
    // (profiled CodeBlock, kind/bytecode/operand key) -> prediction the compile consumed
    HashMap<std::pair<CodeBlock*, uint64_t>, SpeculatedType> consumed;
    bool refuted { false };
    bool jettisonRequested { false };
    bool causeSplitRearm { false };
    HashMap<unsigned, uint64_t> transientTickPlusOne; // code origin hash -> tick + 1
    // work counter (calls + loop back-edges) of this optimized code; tier 0 DFG, 1 FTL
    std::unique_ptr<uint64_t> work;
    unsigned workTier { 0 };
    CodeBlock* workBaseline { nullptr };
    // cause-split persistent site, handed to the baseline on jettison
    bool persistentHasValue { false };
    SpeculatedType persistentValue { SpecNone };
    CodeBlock* persistentProfiled { nullptr };
    BytecodeIndex persistentIndex;
    Vector<std::unique_ptr<ExitSiteInfo>> exitSites; // baked into this code's exit ramps
};

constexpr unsigned kDPs = static_cast<unsigned>(DP::Count);

struct Stats {
    uint64_t tierVotes[NumTierExperts][kDPs][3] { };
    uint64_t deoptVotes[NumDeoptExperts][kDPs][3] { };
    uint64_t acted[kDPs][3] { };
    uint64_t actedDiffersFromJsc[kDPs] { };
    uint64_t pulls { 0 }, quietGaps { 0 };
    uint64_t reevaluations { 0 };
};

// A function's scoring data, kept when the function's CodeBlock is destroyed.
struct Record {
    int8_t functionClass;
    double ms[3];      // sampled ms in Baseline / DFG / FTL
    uint64_t work[3];  // work units in Baseline / DFG / FTL
    RawLoss raw;
};

Lock s_lock;
HashMap<CodeBlock*, std::unique_ptr<FnState>>* s_fn;
HashMap<CodeBlock*, std::unique_ptr<OptState>>* s_opt;
Stats s_stats;
Vector<Record>* s_retired;
HashSet<CodeBlock*>* s_waiters; // functions with pending waiters (waiting[] != 0)
MonotonicTime s_lastReevaluation;
uint64_t s_pointId { 0 };
// quiescence: run-wide running median of broken quiet gaps (ms), two heaps
std::priority_queue<double>* s_gapLow;                                            // max-heap
std::priority_queue<double, std::vector<double>, std::greater<double>>* s_gapHigh; // min-heap
// queue-slack: EWMA of completed compiles' bytecode cost, [0] DFG, [1] FTL
double s_ewma[2] { 0, 0 };
bool s_haveEwma[2] { false, false };
VM* s_vm { nullptr };
bool s_dumped { false };

FnState* findFnState(CodeBlock* baseline) WTF_REQUIRES_LOCK(s_lock)
{
    auto it = s_fn->find(baseline);
    return it == s_fn->end() ? nullptr : it->value.get();
}

FnState& fnState(CodeBlock* baseline) WTF_REQUIRES_LOCK(s_lock)
{
    auto& p = s_fn->add(baseline, nullptr).iterator->value;
    if (!p)
        p = std::make_unique<FnState>();
    return *p;
}

OptState& optState(CodeBlock* optimized) WTF_REQUIRES_LOCK(s_lock)
{
    auto& p = s_opt->add(optimized, nullptr).iterator->value;
    if (!p)
        p = std::make_unique<OptState>();
    return *p;
}

double intervalMs()
{
    return Options::tierUpTimeAttributionIntervalUs() / 1000.0;
}

// small (within the DFG inline budget) / loopy / large (beyond the FTL inline budget) / other.
// Main thread only (reads bytecode).
int8_t classOf(CodeBlock* baseline, FnState& st)
{
    if (st.functionClass >= 0)
        return st.functionClass;
    unsigned cost = baseline->bytecodeCost();
    int8_t c = 3;
    if (baseline->codeType() == FunctionCode && cost <= Options::maximumFunctionForCallInlineCandidateBytecodeCostForDFG())
        c = 0;
    else {
        bool loopy = false;
        for (const auto& instruction : baseline->instructions()) {
            if (instruction->opcodeID() == op_loop_hint) {
                loopy = true;
                break;
            }
        }
        if (loopy)
            c = 1;
        else if (cost > Options::maximumFunctionForCallInlineCandidateBytecodeCostForFTL())
            c = 2;
    }
    st.functionClass = c;
    return c;
}

void closePendingEarly(FnState& st, unsigned step) WTF_REQUIRES_LOCK(s_lock)
{
    if (!st.pendingEarly[step])
        return;
    st.raw.early.append({ static_cast<uint8_t>(step), st.pendingYes[step], st.pendingAwake[step],
        { st.pendingLowerMs[step][0], st.pendingLowerMs[step][1] } });
    st.pendingEarly[step] = false;
    st.pendingLowerMs[step][0] = st.pendingLowerMs[step][1] = 0;
}

CodeBlock* baselineOf(CodeBlock* codeBlock)
{
    CodeBlock* b = codeBlock->baselineAlternative();
    return b ? b : codeBlock;
}

bool isLoopDP(DP dp) { return dp == DP::BaselineLoopOSR || dp == DP::DFGLoopOSR; }
bool isFTLDP(DP dp) { return dp == DP::DFGToFTL || dp == DP::DFGLoopOSR; }

void addGap(double ms) WTF_REQUIRES_LOCK(s_lock)
{
    if (s_gapLow->empty() || ms <= s_gapLow->top())
        s_gapLow->push(ms);
    else
        s_gapHigh->push(ms);
    if (s_gapLow->size() > s_gapHigh->size() + 1) {
        s_gapHigh->push(s_gapLow->top());
        s_gapLow->pop();
    } else if (s_gapHigh->size() > s_gapLow->size()) {
        s_gapLow->push(s_gapHigh->top());
        s_gapHigh->pop();
    }
    s_stats.quietGaps++;
}

std::optional<double> gapMedian() WTF_REQUIRES_LOCK(s_lock)
{
    if (s_gapLow->empty())
        return std::nullopt;
    if (s_gapLow->size() == s_gapHigh->size())
        return (s_gapLow->top() + s_gapHigh->top()) / 2;
    return s_gapLow->top();
}

uint64_t workOf(CodeBlock* baseline, FnState& st, unsigned tierIndex) WTF_REQUIRES_LOCK(s_lock)
{
    if (!tierIndex) {
        auto* jitData = baseline->baselineJITData();
        return jitData ? jitData->minutemanWork() : 0;
    }
    uint64_t w = st.retiredWork[tierIndex - 1];
    for (CodeBlock* optimized : st.liveOptimized) {
        auto it = s_opt->find(optimized);
        if (it != s_opt->end() && it->value->work && it->value->workTier == tierIndex - 1)
            w += *it->value->work;
    }
    return w;
}

Record makeRecord(CodeBlock* baseline, FnState& st) WTF_REQUIRES_LOCK(s_lock)
{
    for (unsigned step = 0; step < 2; ++step)
        closePendingEarly(st, step); // never re-tiered: charged all its lower-tier time so far
    Record r { st.functionClass, { st.sampleMs[0], st.sampleMs[1], st.sampleMs[2] }, { 0, 0, 0 }, WTF::move(st.raw) };
    for (unsigned t = 0; t < 3; ++t)
        r.work[t] = workOf(baseline, st, t);
    st.raw = RawLoss { };
    return r;
}

bool isDeoptDP(unsigned dp)
{
    return dp == static_cast<unsigned>(DP::Exit) || dp == static_cast<unsigned>(DP::LoopReoptimize) || dp == static_cast<unsigned>(DP::Retier);
}

// End of run: price every function's raw losses with this run's measured tier speedups.
// speedup(lo, hi) = (sampled ms per work unit in lo) / (sampled ms per work unit in hi), from the
// function's own samples and work counters, else its class's aggregate, else the loss is
// reported unpriced. Every loss is extra time:
//   late      lower-tier ms x (1 - 1/speedup)
//   delayed   higher-tier ms x (speedup - 1)
//   early     (exit + until-re-tier) lower-tier ms x (1 - 1/speedup)
//   synthetic the mean priced early cost of this function at that step, else of its class
// Regret of expert e = its losses minus jsc-default's losses at the points where e was awake.
void priceAndReport() WTF_REQUIRES_LOCK(s_lock)
{
    Vector<Record> records;
    for (auto& [function, st] : *s_fn)
        records.append(makeRecord(function, *st));
    for (auto& r : *s_retired)
        records.append(WTF::move(r));
    s_retired->clear();

    // pairs: (B,D) (B,F) (D,F)
    constexpr unsigned pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
    double classMs[5][3][2] { }, classWork[5][3][2] { };
    auto cls = [](const Record& r) { return r.functionClass < 0 ? 4u : static_cast<unsigned>(r.functionClass); };
    for (auto& r : records) {
        for (unsigned p = 0; p < 3; ++p) {
            unsigned lo = pairs[p][0], hi = pairs[p][1];
            if (r.ms[lo] > 0 && r.work[lo] && r.ms[hi] > 0 && r.work[hi]) {
                classMs[cls(r)][p][0] += r.ms[lo];
                classWork[cls(r)][p][0] += r.work[lo];
                classMs[cls(r)][p][1] += r.ms[hi];
                classWork[cls(r)][p][1] += r.work[hi];
            }
        }
    }
    unsigned fromFunction = 0, fromClass = 0, missing = 0;
    auto speedup = [&](const Record& r, unsigned p) -> std::optional<double> {
        unsigned lo = pairs[p][0], hi = pairs[p][1];
        if (r.ms[lo] > 0 && r.work[lo] && r.ms[hi] > 0 && r.work[hi]) {
            fromFunction++;
            return (r.ms[lo] / r.work[lo]) / (r.ms[hi] / r.work[hi]);
        }
        unsigned c = cls(r);
        if (classMs[c][p][0] > 0 && classMs[c][p][1] > 0) {
            fromClass++;
            return (classMs[c][p][0] / classWork[c][p][0]) / (classMs[c][p][1] / classWork[c][p][1]);
        }
        missing++;
        return std::nullopt;
    };

    double late[NumTierExperts] { }, delayed[NumTierExperts] { }, early[NumTierExperts] { }, synthetic[NumTierExperts] { };
    double shadow[NumTierExperts] { };
    double unpricedMs = 0;
    unsigned earlyEvents = 0, syntheticEvents = 0, syntheticUnpriced = 0;
    // Pass 1: price jettisons; per-record and per-class mean cost per step (for synthetic losses).
    Vector<std::array<std::pair<double, unsigned>, 2>> recordEarly(records.size());
    std::pair<double, unsigned> classEarly[5][2] { };
    for (size_t i = 0; i < records.size(); ++i) {
        auto& r = records[i];
        recordEarly[i] = { { { 0, 0 }, { 0, 0 } } };
        for (auto& ev : r.raw.early) {
            earlyEvents++;
            double cost = 0;
            bool priced = true;
            if (ev.step == 0) {
                auto s0 = speedup(r, 0);
                if (s0)
                    cost = ev.lowerMs[0] * (1 - 1 / *s0);
                else
                    priced = false;
            } else {
                auto sBF = speedup(r, 1), sDF = speedup(r, 2);
                if (ev.lowerMs[0] > 0) {
                    if (sBF)
                        cost += ev.lowerMs[0] * (1 - 1 / *sBF);
                    else
                        priced = false;
                }
                if (ev.lowerMs[1] > 0) {
                    if (sDF)
                        cost += ev.lowerMs[1] * (1 - 1 / *sDF);
                    else
                        priced = false;
                }
            }
            if (!priced) {
                unpricedMs += ev.lowerMs[0] + ev.lowerMs[1];
                continue;
            }
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if (ev.yes & (1 << e))
                    early[e] += cost;
                if ((ev.yes & (1 << TJsc)) && (ev.awake & (1 << e)))
                    shadow[e] += cost;
            }
            recordEarly[i][ev.step].first += cost;
            recordEarly[i][ev.step].second++;
            classEarly[cls(r)][ev.step].first += cost;
            classEarly[cls(r)][ev.step].second++;
        }
    }
    // Pass 2: late, delayed, synthetic.
    for (size_t i = 0; i < records.size(); ++i) {
        auto& r = records[i];
        auto sBD = speedup(r, 0), sBF = speedup(r, 1), sDF = speedup(r, 2);
        for (unsigned e = 0; e < NumTierExperts; ++e) {
            // late: step 0 Baseline ms (B->D), step 1 DFG ms (D->F)
            if (sBD) {
                late[e] += r.raw.late[0][e] * (1 - 1 / *sBD);
                shadow[e] += r.raw.lateJsc[0][e] * (1 - 1 / *sBD);
            } else
                unpricedMs += r.raw.late[0][e];
            if (sDF) {
                late[e] += r.raw.late[1][e] * (1 - 1 / *sDF);
                shadow[e] += r.raw.lateJsc[1][e] * (1 - 1 / *sDF);
            } else
                unpricedMs += r.raw.late[1][e];
            // delayed: step 0 kept in Baseline instead of DFG (B->D) or FTL (B->F); step 1 kept in DFG instead of FTL
            std::optional<double> factor[2][2] = { { sBD, sBF }, { std::nullopt, sDF } };
            for (unsigned si = 0; si < 2; ++si) {
                for (unsigned h = 0; h < 2; ++h) {
                    double raw = r.raw.delayed[si][h][e], rawJsc = r.raw.delayedJsc[si][h][e];
                    if (!raw && !rawJsc)
                        continue;
                    if (factor[si][h]) {
                        delayed[e] += raw * (*factor[si][h] - 1);
                        shadow[e] += rawJsc * (*factor[si][h] - 1);
                    } else
                        unpricedMs += raw;
                }
            }
            // synthetic: this function's mean priced jettison cost at that step, else its class's
            for (unsigned si = 0; si < 2; ++si) {
                unsigned n = r.raw.synthetic[si][e], nJsc = r.raw.syntheticJsc[si][e];
                if (!n && !nJsc)
                    continue;
                syntheticEvents += n;
                std::optional<double> mean;
                if (recordEarly[i][si].second)
                    mean = recordEarly[i][si].first / recordEarly[i][si].second;
                else if (classEarly[cls(r)][si].second)
                    mean = classEarly[cls(r)][si].first / classEarly[cls(r)][si].second;
                if (!mean) {
                    syntheticUnpriced += n;
                    continue;
                }
                synthetic[e] += n * *mean;
                shadow[e] += nJsc * *mean;
            }
        }
    }
    dataLog("tierUpExperts: ghostRegretMs interval=", intervalMs(), " functions=", records.size(),
        " earlyEvents=", earlyEvents, " syntheticEvents=", syntheticEvents, " syntheticUnpriced=", syntheticUnpriced,
        " unpricedMs=", unpricedMs, " speedupFromFunction=", fromFunction, " fromClass=", fromClass, " missing=", missing,
        " reevaluations=", s_stats.reevaluations);
    for (unsigned e = 0; e < NumTierExperts; ++e) {
        double own = late[e] + delayed[e] + early[e] + synthetic[e];
        dataLog(" | ", kTierNames[e], " late/delayed/early/synthetic=", late[e], "/", delayed[e], "/", early[e], "/", synthetic[e],
            " jscAtAwake=", shadow[e], " regret=", own - shadow[e]);
    }
    dataLogLn();
}

void dump()
{
    if (s_dumped || !g_enabled || !Options::verboseTierUpExperts())
        return;
    s_dumped = true;
    Locker locker { s_lock };
    dataLogLn("tierUpExperts: selected tierUp=", kTierNames[s_tierSelected], " deopt=", kDeoptNames[s_deoptSelected],
        " pulls=", s_stats.pulls, " quietGaps=", s_stats.quietGaps);
    priceAndReport();
    for (unsigned dp = 0; dp < kDPs; ++dp) {
        bool deopt = isDeoptDP(dp);
        unsigned n = deopt ? static_cast<unsigned>(NumDeoptExperts) : static_cast<unsigned>(NumTierExperts);
        uint64_t seen = 0;
        for (unsigned v = 0; v < 3; ++v)
            seen += deopt ? s_stats.deoptVotes[0][dp][v] : s_stats.tierVotes[0][dp][v];
        if (!seen)
            continue;
        dataLog("tierUpExperts: dp=", kDPNames[dp], " points=", seen, " acted yes/no=", s_stats.acted[dp][Yes], "/", s_stats.acted[dp][No],
            " differsFromJsc=", s_stats.actedDiffersFromJsc[dp]);
        for (unsigned e = 0; e < n; ++e) {
            const uint64_t* c = deopt ? s_stats.deoptVotes[e][dp] : s_stats.tierVotes[e][dp];
            dataLog(" | ", deopt ? kDeoptNames[e] : kTierNames[e], " yes/no/abstain=", c[Yes], "/", c[No], "/", c[Abstain]);
        }
        dataLogLn();
    }
}

// ---------------------------------------------------------------- expert helpers
unsigned inlineBudget(bool ftl)
{
    return ftl ? Options::maximumFunctionForCallInlineCandidateBytecodeCostForFTL() : Options::maximumFunctionForCallInlineCandidateBytecodeCostForDFG();
}

bool isThin(CodeBlock* callee)
{
    JITType type = callee->jitType();
    if (type == JITType::InterpreterThunk)
        return true;
    if (type != JITType::BaselineJIT)
        return false;
    auto* jitData = callee->baselineJITData();
    if (!jitData)
        return false;
    bool anyCache = false;
    for (auto& cache : jitData->propertyInlineCaches()) {
        anyCache = true;
        if (cache.everConsidered)
            return false;
    }
    return anyCache;
}

// Computed outside s_lock (takes the CodeBlock lock).
uint64_t profileFingerprint(CodeBlock* baseline)
{
    baseline->updatePredictionsConcurrently(); // the same update JSC performs at its own checks
    return baseline->minutemanProfileFingerprint();
}

// Free compile slots for the tier this decision would enqueue into.
int64_t idleCompilerSlots(bool ftl)
{
    JITWorklist* worklist = JITWorklist::existingGlobalWorklistOrNull();
    if (!worklist)
        return 1;
    return worklist->idleCapacity(ftl ? JITPlan::Tier::FTL : JITPlan::Tier::DFG);
}

SpeculatedType specOfEncoded(EncodedJSValue encoded)
{
    JSValue v = JSValue::decode(encoded);
    return v ? speculationFromValue(v) : SpecNone;
}

uint64_t consumedKey(ProfileKind kind, BytecodeIndex index, uint64_t operandBits)
{
    return (static_cast<uint64_t>(kind) << 60) ^ (static_cast<uint64_t>(index.asBits()) << 24) ^ (operandBits & 0xffffff);
}

// The value an exit wrote back, and where the compile's consumed prediction for it is keyed.
struct ExitValue {
    bool have { false };
    SpeculatedType value { SpecNone };
    CodeBlock* profiled { nullptr };
    uint64_t key { 0 };
    BytecodeIndex index;
};

ExitValue exitValue(CodeBlock* optimized, const MethodOfGettingAValueProfile* method)
{
    ExitValue r;
    if (!method || !*method)
        return r;
    CodeBlock* profiled = baselineCodeBlockForOriginAndBaselineCodeBlock(MinutemanValueProfileAccess::codeOrigin(*method), optimized->baselineAlternative());
    BytecodeIndex index = MinutemanValueProfileAccess::codeOrigin(*method).bytecodeIndex();
    switch (MinutemanValueProfileAccess::kind(*method)) {
    case MinutemanValueProfileAccess::Kind::BytecodeValueProfile: {
        JSValue* bucket = profiled->lazyValueProfiles().addSpeculationFailureValueProfile(index);
        r.value = *bucket ? speculationFromValue(*bucket) : SpecNone;
        r.key = consumedKey(ProfileKind::Bytecode, index, 0);
        break;
    }
    case MinutemanValueProfileAccess::Kind::ArgumentValueProfile: {
        Operand operand = Operand::fromBits(MinutemanValueProfileAccess::rawOperand(*method));
        r.value = specOfEncoded(*profiled->valueProfileForArgument(operand.toArgument()).specFailBucket(0));
        r.key = consumedKey(ProfileKind::Argument, BytecodeIndex(0), operand.toArgument());
        break;
    }
    case MinutemanValueProfileAccess::Kind::LazyOperandValueProfile: {
        LazyOperandValueProfileKey key(index, Operand::fromBits(MinutemanValueProfileAccess::rawOperand(*method)));
        LazyOperandValueProfile* profile = profiled->lazyValueProfiles().addOperandValueProfile(key);
        r.value = specOfEncoded(*profile->specFailBucket(0));
        r.key = consumedKey(ProfileKind::Lazy, index, MinutemanValueProfileAccess::rawOperand(*method));
        break;
    }
    default:
        return r; // arith profiles: those exits (Overflow / NegativeZero) are decided by their exit site
    }
    r.have = r.value != SpecNone;
    r.profiled = profiled;
    r.index = index;
    return r;
}

bool isPersistentKind(ExitKind kind)
{
    switch (kind) {
    case BadType:
    case BadConstantValue:
    case BadExecutable:
    case BadCache:
    case BadConstantCache:
    case BadIndexingType:
        return true;
    default:
        return false;
    }
}

bool isTransientKind(ExitKind kind)
{
    switch (kind) {
    case Overflow:
    case NegativeZero:
    case OutOfBounds:
    case Uncountable:
    case InadequateCoverage:
        return true;
    default:
        return false;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------- lifecycle
void initialize(VM& vm)
{
    if (s_vm)
        return;
    auto is = [](const char8_t* option, const char* name) { return option && !strcmp(reinterpret_cast<const char*>(option), name); };
    const char8_t* t = Options::tierUpExpert();
    const char8_t* d = Options::deoptExpert();
    if (!t && !d)
        return;
    s_vm = &vm;
    s_fn = new HashMap<CodeBlock*, std::unique_ptr<FnState>>();
    s_opt = new HashMap<CodeBlock*, std::unique_ptr<OptState>>();
    s_retired = new Vector<Record>();
    s_waiters = new HashSet<CodeBlock*>();
    s_gapLow = new std::priority_queue<double>();
    s_gapHigh = new std::priority_queue<double, std::vector<double>, std::greater<double>>();
    bool found = !t;
    for (unsigned e = 0; t && e < NumTierExperts; ++e) {
        if (is(t, kTierNames[e])) {
            s_tierSelected = e;
            found = true;
        }
    }
    RELEASE_ASSERT_WITH_MESSAGE(found, "tierUpExperts: unknown --tierUpExpert");
    found = !d;
    for (unsigned e = 0; d && e < NumDeoptExperts; ++e) {
        if (is(d, kDeoptNames[e])) {
            s_deoptSelected = e;
            found = true;
        }
    }
    RELEASE_ASSERT_WITH_MESSAGE(found, "tierUpExperts: unknown --deoptExpert");
    RELEASE_ASSERT_WITH_MESSAGE(Options::tierUpTimeAttribution(), "tierUpExperts: the experts need --tierUpTimeAttribution=1");
    g_enabled = true;
    if (Options::verboseTierUpExperts())
        std::atexit(dump);
}

void shutdown(VM& vm)
{
    if (s_vm == &vm)
        dump();
}

void forget(CodeBlock* codeBlock)
{
    Locker locker { s_lock };
    auto it = s_fn->find(codeBlock);
    if (it != s_fn->end()) {
        FnState& st = *it->value;
        for (CodeBlock* caller : st.trackedBy) {
            auto c = s_fn->find(caller);
            if (c == s_fn->end())
                continue;
            for (auto& slot : c->value->slots) {
                if (slot == codeBlock)
                    slot = nullptr;
            }
        }
        for (CodeBlock* callee : st.slots) {
            if (!callee)
                continue;
            auto c = s_fn->find(callee);
            if (c != s_fn->end())
                c->value->trackedBy.remove(codeBlock);
        }
        for (CodeBlock* inlinee : st.inlinees) {
            if (FnState* i = findFnState(inlinee))
                i->inlinedInto.remove(codeBlock);
        }
        for (CodeBlock* inliner : st.inlinedInto) {
            if (FnState* i = findFnState(inliner))
                i->inlinees.removeAll(codeBlock);
        }
        s_retired->append(makeRecord(codeBlock, st)); // keep its scoring data for the end-of-run pricing
        s_waiters->remove(codeBlock);
        s_fn->remove(it);
    }
    if (auto o = s_opt->find(codeBlock); o != s_opt->end()) {
        OptState& os = *o->value;
        if (os.work) {
            if (FnState* b = findFnState(os.workBaseline)) {
                b->retiredWork[os.workTier] += *os.work;
                b->liveOptimized.remove(codeBlock);
            }
        }
        s_opt->remove(o);
    }
}

// ---------------------------------------------------------------- tier-up
// The pure (state-preserving) vote of tier-up expert `e` on `function` at step `si`, used to
// re-evaluate experts that voted Wait at an acted Go. jsc-default's rule is a counter crossing,
// which cannot be re-asked once the function has compiled: a waiting jsc-default stays waiting.
Vote pureVote(unsigned e, unsigned si, CodeBlock* function, FnState& st, uint64_t fingerprint, bool heavy, JITType top,
    int64_t idle, uint64_t tick, MonotonicTime now) WTF_REQUIRES_LOCK(s_lock)
{
    switch (e) {
    case TSupply: {
        unsigned budget = inlineBudget(si == 1);
        bool tracked = false;
        for (unsigned i = 0; i < kSupplySlots; ++i) {
            CodeBlock* callee = st.slots[i];
            if (!callee || callee->bytecodeCost() > budget)
                continue;
            tracked = true;
            if (!st.deferred[i] && isThin(callee))
                return No;
        }
        return tracked ? Yes : Abstain;
    }
    case TQuiescence: {
        auto median = gapMedian();
        if (!median)
            return Abstain;
        if (!st.haveFingerprint || st.fingerprint != fingerprint || st.requireChange)
            return No;
        return (now - st.lastChange).milliseconds() > *median ? Yes : No;
    }
    case TSampleGravity:
        return heavy && top != JITType::FTLJIT ? Yes : No;
    case TQueueSlack: {
        bool recentJettison = st.lastJettisonTickPlusOne && tick + 1 - st.lastJettisonTickPlusOne <= 1;
        if (idle > 0)
            return recentJettison ? Abstain : Yes;
        return s_haveEwma[si] && function->bytecodeCost() <= s_ewma[si] ? Yes : No;
    }
    default:
        return No;
    }
}

// At most once per ms of wall time, at a decision point anywhere: re-ask every waiting expert.
void reevaluateWaiters()
{
    MonotonicTime now = MonotonicTime::now();
    Vector<CodeBlock*> functions;
    {
        Locker locker { s_lock };
        if ((now - s_lastReevaluation).milliseconds() < 1 || s_waiters->isEmpty())
            return;
        s_lastReevaluation = now;
        functions = copyToVector(*s_waiters);
    }
    int64_t idle[2] = { idleCompilerSlots(false), idleCompilerSlots(true) };
    uint64_t tick = TierUpTimeAttribution::currentTick();
    for (CodeBlock* function : functions) {
        uint64_t fingerprint = profileFingerprint(function);
        bool heavy = TierUpTimeAttribution::inLowerTierHeavySet(function);
        CodeBlock* replacement = function->replacement();
        JITType top = replacement ? replacement->jitType() : function->jitType();
        Locker locker { s_lock };
        FnState* st = findFnState(function);
        if (!st)
            continue;
        for (unsigned si = 0; si < 2; ++si) {
            for (unsigned e = 1; e < NumTierExperts; ++e) {
                if (!(st->waiting[si] & (1 << e)))
                    continue;
                s_stats.reevaluations++;
                if (pureVote(e, si, function, *st, fingerprint, heavy, top, idle[si], tick, now) != No)
                    st->waiting[si] &= ~(1 << e);
            }
        }
        if (!st->waiting[0] && !st->waiting[1])
            s_waiters->remove(function);
    }
}

bool tierUp(DP dp, CodeBlock* owner, bool jscGo)
{
    reevaluateWaiters();
    CodeBlock* baseline = baselineOf(owner);
    unsigned dpi = static_cast<unsigned>(dp);
    bool ftl = isFTLDP(dp);
    bool lb = dp == DP::LLIntToBaseline;

    // Inputs gathered outside s_lock.
    uint64_t fingerprint = lb ? 0 : profileFingerprint(baseline);
    // Ghost scoring: the sites a ghost compile would have used are approximated by this
    // function's own profiles plus those of the inlinees of its most recent real compile
    // (approved approximation (a), validated by the ghost-vs-live ranking check).
    uint64_t scoringFingerprint = fingerprint;
    if (!lb) {
        Vector<CodeBlock*> inlinees;
        {
            Locker locker { s_lock };
            if (FnState* st = findFnState(baseline))
                inlinees = st->inlinees;
        }
        for (CodeBlock* inlinee : inlinees)
            scoringFingerprint = WTF::pairIntHash(static_cast<unsigned>(scoringFingerprint), static_cast<unsigned>(profileFingerprint(inlinee))) ^ (scoringFingerprint << 1);
    }
    MonotonicTime now = MonotonicTime::now();
    bool heavy = !lb && TierUpTimeAttribution::inLowerTierHeavySet(baseline) && owner->jitType() != JITType::FTLJIT;
    int64_t idle = lb ? 0 : idleCompilerSlots(ftl);
    uint64_t tick = TierUpTimeAttribution::currentTick();
    unsigned cost = baseline->bytecodeCost();

    Vote v[NumTierExperts] { };
    Vote retier = Abstain;
    CodeBlock* pullTarget = nullptr;
    Vote acted;
    {
        Locker locker { s_lock };
        FnState& st = fnState(baseline);

        v[TJsc] = jscGo ? Yes : No;

        // supply
        if (lb)
            v[TSupply] = st.pulled ? Yes : Abstain;
        else if (!isLoopDP(dp)) {
            unsigned budget = inlineBudget(ftl);
            bool tracked = false;
            for (unsigned i = 0; i < kSupplySlots && v[TSupply] != No; ++i) {
                CodeBlock* callee = st.slots[i];
                if (!callee || callee->bytecodeCost() > budget)
                    continue;
                tracked = true;
                if (!st.deferred[i] && isThin(callee)) {
                    st.deferred[i] = true;
                    fnState(callee).pulled = true;
                    pullTarget = callee;
                    v[TSupply] = No;
                }
            }
            if (v[TSupply] != No)
                v[TSupply] = tracked ? Yes : Abstain;
        }

        // quiescence
        if (!lb) {
            bool changed = !st.haveFingerprint || st.fingerprint != fingerprint;
            if (changed) {
                if (st.haveFingerprint) {
                    addGap((now - st.lastChange).milliseconds());
                    st.requireChange = false;
                }
                st.fingerprint = fingerprint;
                st.haveFingerprint = true;
                st.lastChange = now;
            }
            auto median = gapMedian();
            if (!median)
                v[TQuiescence] = Abstain;
            else if (changed || st.requireChange)
                v[TQuiescence] = No;
            else
                v[TQuiescence] = (now - st.lastChange).milliseconds() > *median ? Yes : No;
        }

        // sample-gravity
        if (!lb)
            v[TSampleGravity] = heavy ? Yes : No;

        // queue-slack
        if (!lb) {
            bool recentJettison = st.lastJettisonTickPlusOne && tick + 1 - st.lastJettisonTickPlusOne <= 1;
            if (idle > 0)
                v[TQueueSlack] = recentJettison ? Abstain : Yes;
            else {
                unsigned tier = ftl ? 1 : 0;
                bool small = s_haveEwma[tier] && cost <= s_ewma[tier];
                v[TQueueSlack] = (small || isLoopDP(dp)) ? Yes : No;
            }
        }

        // cause-split retier (deopt family)
        bool isRetier = st.retierPending && (dp == DP::BaselineToDFG || dp == DP::BaselineLoopOSR);
        if (isRetier) {
            if (!st.retierHasValue)
                retier = Yes; // nothing to wait for (e.g. an inline-cache exit without a value profile)
            else {
                SpeculatedType current = st.retierProfiled->valueProfilePredictionForBytecodeIndex(st.retierIndex);
                retier = (current & st.retierValue) == st.retierValue ? Yes : Abstain;
            }
            unsigned r = static_cast<unsigned>(DP::Retier);
            s_stats.deoptVotes[DJsc][r][v[TJsc]]++;
            s_stats.deoptVotes[DRefutation][r][Abstain]++;
            s_stats.deoptVotes[DCauseSplit][r][retier]++;
        }

        for (unsigned e = 0; e < NumTierExperts; ++e)
            s_stats.tierVotes[e][dpi][v[e]]++;
        acted = v[s_tierSelected] == Abstain ? v[TJsc] : v[s_tierSelected];

        // ghost scoring
        uint64_t point = ++s_pointId;
        if (!lb) {
            unsigned si = ftl ? 1 : 0;
            classOf(baseline, st);
            // Un-acted Go votes whose profile has since changed: their compile would have exited.
            // Priced at the end of the run from this run's real jettison costs.
            uint8_t superseded = 0;
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if (st.ghostActive[si][e] && st.ghostFingerprint[si][e] != scoringFingerprint) {
                    superseded |= 1 << e;
                    st.ghostActive[si][e] = false;
                }
            }
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if (!(superseded & (1 << e)))
                    continue;
                st.raw.synthetic[si][e]++;
                if ((superseded & (1 << TJsc)) && st.ghostPoint[si][TJsc] == st.ghostPoint[si][e])
                    st.raw.syntheticJsc[si][e]++;
            }
            uint8_t awake = 0;
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                st.latest[si][e] = v[e];
                if (v[e] != Abstain)
                    awake |= 1 << e;
            }
            st.latestAwake[si] = awake;
            st.haveLatest[si] = true;
        }
        if (isRetier) {
            unsigned r = static_cast<unsigned>(DP::Retier);
            Vote retierActed = (s_deoptSelected == DCauseSplit && retier == Yes) ? Yes : acted;
            s_stats.acted[r][retierActed]++;
            if ((retierActed == Yes) != jscGo)
                s_stats.actedDiffersFromJsc[r]++;
            acted = retierActed;
        }
        if (acted == Yes) {
            if (isRetier)
                st.retierPending = false;
            if (lb)
                st.pulled = false;
        }
        if (!lb) {
            unsigned si = ftl ? 1 : 0;
            if (acted == Yes) {
                uint8_t yes = 0, wait = 0;
                for (unsigned e = 0; e < NumTierExperts; ++e) {
                    if (v[e] == Yes)
                        yes |= 1 << e;
                    else if (v[e] == No)
                        wait |= 1 << e;
                    st.ghostActive[si][e] = false; // the real compile supersedes every ghost
                }
                st.compileYesMask[si] = yes;
                st.compileAwakeMask[si] = st.latestAwake[si];
                // Experts that voted Wait are charged the extra time they would have kept the
                // function in the lower tier, until their re-evaluated vote turns Go.
                st.waiting[si] = wait;
                st.waitingAwake[si] = st.latestAwake[si];
                if (wait)
                    s_waiters->add(baseline);
            } else {
                for (unsigned e = 0; e < NumTierExperts; ++e) {
                    if (v[e] == Yes && !st.ghostActive[si][e]) {
                        st.ghostActive[si][e] = true;
                        st.ghostFingerprint[si][e] = scoringFingerprint;
                        st.ghostPoint[si][e] = point;
                    }
                }
            }
        }
        s_stats.acted[dpi][acted]++;
        if ((acted == Yes) != jscGo)
            s_stats.actedDiffersFromJsc[dpi]++;
        bool supplyActed = s_tierSelected == TSupply && v[TSupply] == No && acted == No;
        if (supplyActed && pullTarget)
            s_stats.pulls++;
        else
            pullTarget = nullptr;
    }

    if (pullTarget)
        pullTarget->jitNextInvocation(); // the pulled callee's next LLInt check is now
    if (acted == No && jscGo) {
        // The acting expert waits where JSC would have gone: re-arm the way JSC itself re-arms
        // when its own gate says wait (shouldOptimizeNowFromBaseline -> optimizeAfterWarmUp).
        // (No expert votes Wait at LLInt->Baseline; jitSoon is JSC's re-arm there.)
        if (lb)
            owner->jitSoon();
        else if (ftl)
            owner->jitCode()->dfg()->optimizeAfterWarmUp(owner);
        else
            owner->optimizeAfterWarmUp();
    }
    return acted == Yes;
}

// ---------------------------------------------------------------- deopt
const ExitSiteInfo* retainExitSiteInfo(CodeBlock* optimized, const DFG::OSRExitBase& exit, const MethodOfGettingAValueProfile& valueProfile)
{
    auto info = std::unique_ptr<ExitSiteInfo>(new ExitSiteInfo { exit.m_kind, exit.m_codeOrigin, exit.m_codeOriginForExitProfile, valueProfile });
    const ExitSiteInfo* result = info.get();
    Locker locker { s_lock };
    optState(optimized).exitSites.append(WTF::move(info));
    return result;
}

bool exitDecision(CodeBlock* optimized, const ExitSiteInfo* exit)
{
    const MethodOfGettingAValueProfile* method = &exit->valueProfile;
    CodeBlock* baseline = optimized->baselineAlternative();
    ExitKind kind = exit->kind;
    CodeOrigin profileOrigin = exit->codeOriginForExitProfile;
    CodeBlock* profiled = baselineCodeBlockForOriginAndBaselineCodeBlock(profileOrigin, baseline);
    BytecodeIndex index = profileOrigin.bytecodeIndex();
    bool siteKnown = profiled->unlinkedCodeBlock()->hasExitSite(DFG::FrequentExitSite(index, kind));
    ExitValue ev = exitValue(optimized, method);
    uint64_t tick = TierUpTimeAttribution::currentTick();

    // jsc-default: the exit ramp's test, then operationTriggerReoptimizationNow's own test.
    bool inlinedLoops = false;
    for (InlineCallFrame* frame = exit->codeOrigin.inlineCallFrame(); frame; frame = frame->directCaller.inlineCallFrame()) {
        if (frame->baselineCodeBlock->ownerExecutable()->didTryToEnterInLoop())
            inlinedLoops = true;
    }
    auto* jitData = baseline->baselineJITData();
    bool hotBaseline = jitData && jitData->executeCounter().m_counter >= 0;
    uint32_t exits = optimized->osrExitCounter();
    uint32_t rampThreshold = inlinedLoops ? optimized->exitCountThresholdForReoptimizationFromLoop() : optimized->exitCountThresholdForReoptimization();
    CodeBlock* replacement = baseline->replacement();
    bool jsc = false;
    if ((hotBaseline || exits > rampThreshold) && replacement && replacement != baseline) {
        bool didExitABunch = exits >= optimized->exitCountThresholdForReoptimization();
        jsc = didExitABunch || ((baseline->checkIfOptimizationThresholdReached() || inlinedLoops) && exits >= optimized->exitCountThresholdForReoptimizationFromLoop());
    }

    Locker locker { s_lock };
    OptState& os = optState(optimized);
    Vote v[NumDeoptExperts] { };
    v[DJsc] = jsc ? Yes : No;

    // refutation
    bool refuted = !siteKnown;
    if (!refuted && ev.have) {
        auto it = os.consumed.find(std::pair { ev.profiled, ev.key });
        if (it != os.consumed.end() && (ev.value & ~it->value))
            refuted = true;
    }
    if (refuted)
        os.refuted = true;
    v[DRefutation] = refuted ? Yes : No;

    // cause-split
    if (isPersistentKind(kind))
        v[DCauseSplit] = siteKnown ? Abstain : Yes;
    else if (isTransientKind(kind)) {
        auto& last = os.transientTickPlusOne.add(exit->codeOrigin.hash(), 0).iterator->value;
        v[DCauseSplit] = (last && last != tick + 1) ? Yes : No;
        last = tick + 1;
    } else
        v[DCauseSplit] = Abstain;

    unsigned dpi = static_cast<unsigned>(DP::Exit);
    for (unsigned e = 0; e < NumDeoptExperts; ++e)
        s_stats.deoptVotes[e][dpi][v[e]]++;
    Vote acted = v[s_deoptSelected] == Abstain ? v[DJsc] : v[s_deoptSelected];
    s_stats.acted[dpi][acted]++;
    if ((acted == Yes) != jsc)
        s_stats.actedDiffersFromJsc[dpi]++;
    if (acted == Yes) {
        os.jettisonRequested = true;
        if (s_deoptSelected == DCauseSplit && v[DCauseSplit] == Yes && isPersistentKind(kind)) {
            os.causeSplitRearm = true;
            os.persistentHasValue = ev.have;
            os.persistentValue = ev.value;
            os.persistentProfiled = ev.have ? ev.profiled : profiled;
            os.persistentIndex = ev.have ? ev.index : index;
        }
    }
    return acted == Yes;
}

bool takeJettisonRequest(CodeBlock* optimized)
{
    Locker locker { s_lock };
    auto it = s_opt->find(optimized);
    if (it == s_opt->end())
        return false;
    return std::exchange(it->value->jettisonRequested, false);
}

bool loopReoptimize(CodeBlock* optimized, bool jscJettison)
{
    Locker locker { s_lock };
    OptState& os = optState(optimized);
    Vote v[NumDeoptExperts] { };
    v[DJsc] = jscJettison ? Yes : No;
    v[DRefutation] = os.refuted ? Yes : No;
    v[DCauseSplit] = Abstain;
    unsigned dpi = static_cast<unsigned>(DP::LoopReoptimize);
    for (unsigned e = 0; e < NumDeoptExperts; ++e)
        s_stats.deoptVotes[e][dpi][v[e]]++;
    Vote acted = v[s_deoptSelected] == Abstain ? v[DJsc] : v[s_deoptSelected];
    s_stats.acted[dpi][acted]++;
    if ((acted == Yes) != jscJettison)
        s_stats.actedDiffersFromJsc[dpi]++;
    return acted == Yes;
}

// ---------------------------------------------------------------- observations
void noteIncomingCall(CodeBlock* callee, JSCell* caller)
{
    CodeBlock* callerCodeBlock = dynamicDowncast<CodeBlock>(caller);
    if (!callerCodeBlock || callerCodeBlock == callee)
        return;
    if (callee->jitType() != JITType::InterpreterThunk || callee->codeType() != FunctionCode || callee->bytecodeCost() > inlineBudget(true))
        return;
    CodeBlock* callerBaseline = baselineOf(callerCodeBlock);
    Locker locker { s_lock };
    FnState& c = fnState(callerBaseline);
    for (CodeBlock* slot : c.slots) {
        if (slot == callee)
            return;
    }
    unsigned i = c.nextSlot++ % kSupplySlots;
    if (CodeBlock* old = c.slots[i]) {
        auto o = s_fn->find(old);
        if (o != s_fn->end())
            o->value->trackedBy.remove(callerBaseline);
    }
    c.slots[i] = callee;
    c.deferred[i] = false;
    fnState(callee).trackedBy.add(callerBaseline);
}

void noteOptimizedInstalled(CodeBlock* machine)
{
    unsigned tier = machine->jitType() == JITType::FTLJIT ? 1 : 0;
    double cost = machine->bytecodeCost();
    Vector<CodeBlock*> inlined;
    if (auto* common = machine->jitCode() ? machine->jitCode()->dfgCommon() : nullptr; common && common->inlineCallFrames) {
        for (InlineCallFrame* frame : *common->inlineCallFrames) {
            CodeBlock* b = frame->baselineCodeBlock.get();
            if (b && !inlined.contains(b))
                inlined.append(b);
        }
    }
    CodeBlock* baseline = baselineOf(machine);
    Locker locker { s_lock };
    {
        FnState& st = fnState(baseline);
        classOf(baseline, st);
        // Re-tiered: a jettison at this step is closed with the time it cost.
        closePendingEarly(st, tier);
        st.installed[tier] = true;
        st.installedLowerMs[tier][0] = st.installedLowerMs[tier][1] = 0;
        if (tier == 1)
            st.installed[0] = false; // DFG code superseded by FTL: Baseline time now belongs to the FTL compile
        for (CodeBlock* old : st.inlinees) {
            if (FnState* i = findFnState(old))
                i->inlinedInto.remove(baseline);
        }
        st.inlinees = WTF::move(inlined);
        for (CodeBlock* inlinee : st.inlinees)
            fnState(inlinee).inlinedInto.add(baseline);
    }
    if (!s_haveEwma[tier]) {
        s_ewma[tier] = cost;
        s_haveEwma[tier] = true;
    } else
        s_ewma[tier] += kEwmaWeight * (cost - s_ewma[tier]);
}

void noteJettison(CodeBlock* optimized, Profiler::JettisonReason reason)
{
    CodeBlock* baseline = baselineOf(optimized);
    uint64_t tick = TierUpTimeAttribution::currentTick();
    bool rearm = false;
    {
        Locker locker { s_lock };
        FnState& st = fnState(baseline);
        st.lastJettisonTickPlusOne = tick + 1;
        unsigned si = optimized->jitType() == JITType::FTLJIT ? 1 : 0;
        if (Profiler::isSpeculationFailure(reason)) {
            st.requireChange = true;
            if (st.compileYesMask[si] && !st.pendingEarly[si]) {
                // Its cost: the exits' lower-tier time while installed, plus the lower-tier time
                // from now until the function re-tiers (closed at the next install at this step).
                st.pendingEarly[si] = true;
                st.pendingYes[si] = st.compileYesMask[si];
                st.pendingAwake[si] = st.compileAwakeMask[si];
                st.pendingLowerMs[si][0] = st.installedLowerMs[si][0];
                st.pendingLowerMs[si][1] = st.installedLowerMs[si][1];
            }
            st.compileYesMask[si] = 0;
        }
        st.installed[si] = false;
        st.installedLowerMs[si][0] = st.installedLowerMs[si][1] = 0;
        // The jettisoned compile is gone: its waiters' counterfactual ends with it.
        st.waiting[si] = 0;
        if (!st.waiting[0] && !st.waiting[1])
            s_waiters->remove(baseline);
        auto it = s_opt->find(optimized);
        if (it != s_opt->end() && it->value->causeSplitRearm) {
            OptState& os = *it->value;
            os.causeSplitRearm = false;
            st.retierPending = true;
            st.retierHasValue = os.persistentHasValue;
            st.retierValue = os.persistentValue;
            st.retierProfiled = os.persistentProfiled;
            st.retierIndex = os.persistentIndex;
            rearm = true;
        }
    }
    if (rearm)
        baseline->optimizeSoon(); // Cause-Split: "check again soon" instead of JSC's warm-up
}

void noteSample(CodeBlock* root, unsigned tier)
{
    // tier as TierUpTimeAttribution::Tier: 0 LLInt, 1 Baseline, 2 DFG, 3 FTL, 4 native.
    if (tier < 1 || tier > 3)
        return;
    double ms = intervalMs();
    Locker locker { s_lock };
    FnState* st = findFnState(root);
    if (!st)
        return;
    unsigned t = tier - 1; // 0 Baseline, 1 DFG, 2 FTL
    st->sampleMs[t] += ms;
    // Jettison cost accounting: lower tiers are Baseline for step 0, Baseline and DFG for step 1.
    for (unsigned si = 0; si < 2; ++si) {
        if (t > si)
            continue;
        if (st->installed[si])
            st->installedLowerMs[si][t] += ms;
        if (st->pendingEarly[si])
            st->pendingLowerMs[si][t] += ms;
    }
    // Late: a Baseline sample scores step 0, a DFG sample step 1, against the latest votes.
    if (t <= 1 && st->haveLatest[t]) {
        unsigned si = t;
        bool anyGo = false;
        for (unsigned e = 0; e < NumTierExperts; ++e)
            anyGo |= st->latest[si][e] == Yes;
        if (anyGo) {
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if (st->latest[si][e] == No)
                    st->raw.late[si][e] += ms;
                if (st->latest[si][TJsc] == No && (st->latestAwake[si] & (1 << e)))
                    st->raw.lateJsc[si][e] += ms;
            }
        }
    }
    // Delayed: higher-tier time while an expert that voted Wait at the acted Go still waits.
    for (unsigned si = 0; si < 2; ++si) {
        if (!st->waiting[si] || t <= si)
            continue;
        unsigned h = t - 1; // 0 DFG, 1 FTL
        for (unsigned e = 0; e < NumTierExperts; ++e) {
            if (st->waiting[si] & (1 << e))
                st->raw.delayed[si][h][e] += ms;
            if ((st->waiting[si] & (1 << TJsc)) && (st->waitingAwake[si] & (1 << e)))
                st->raw.delayedJsc[si][h][e] += ms;
        }
    }
}

uint64_t* workCounterFor(CodeBlock* machine, bool ftl)
{
    CodeBlock* baseline = baselineOf(machine);
    Locker locker { s_lock };
    OptState& os = optState(machine);
    if (!os.work) {
        os.work = std::make_unique<uint64_t>(0);
        os.workTier = ftl ? 1 : 0;
        os.workBaseline = baseline;
        fnState(baseline).liveOptimized.add(machine);
    }
    return os.work.get();
}

void noteConsumedPrediction(CodeBlock* machine, CodeBlock* profiled, ProfileKind kind, BytecodeIndex index, uint64_t operandBits, SpeculatedType prediction)
{
    Locker locker { s_lock };
    auto& slot = optState(machine).consumed.add(std::pair { profiled, consumedKey(kind, index, operandBits) }, SpecNone).iterator->value;
    slot |= prediction;
}

} // namespace TierUpExperts
} // namespace JSC

#endif // ENABLE(DFG_JIT)
