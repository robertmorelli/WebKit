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
bool g_deoptEnabled { false };

namespace {

// ---------------------------------------------------------------- fixed constants
constexpr unsigned kSupplySlots = 4;          // Caller-Pull: tracked cold inlinable callees per caller
constexpr double kEwmaWeight = 1.0 / 8;       // Queue-Slack: Jacobson's TCP RTT gain

enum TierExpert : unsigned { TJsc, TSupply, TQuiescence, TSampleGravity, TQueueSlack, TDfgProbe, TTierInversion, NumTierExperts };
enum DeoptExpert : unsigned { DJsc, DRefutation, DCauseSplit, NumDeoptExperts };
enum Vote : uint8_t { Abstain = 0, Yes = 1, No = 2 }; // Yes = Go / Jettison, No = Wait / Keep

const char* const kTierNames[] = { "jsc-default", "supply", "quiescence", "sample-gravity", "queue-slack", "dfg-probe", "tier-inversion", "mixer" };
const char* const kDeoptNames[] = { "jsc-default", "refutation", "cause-split" };
const char* const kDPNames[] = { "LLIntToBaseline", "BaselineToDFG", "BaselineLoopOSR", "DFGToFTL", "DFGLoopOSR", "Retier", "Exit", "LoopReoptimize" };

unsigned s_tierSelected { TJsc };
constexpr unsigned TMixer = NumTierExperts; // s_tierSelected value for the mixer
uint8_t s_mix { 0xff };  // tier-up experts that compute votes (lean build: only the mix's)
constexpr double kLearningRate = 0.45; // LeCaR's learning rate
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
    double late[2][NumTierExperts] { };          // lower-tier ms while waiting after some expert voted Go
    double lateJsc[2][NumTierExperts] { };
    double delayed[2][2][NumTierExperts] { };    // [step][higher tier: DFG, FTL][e] higher-tier ms while still waiting
    double delayedJsc[2][2][NumTierExperts] { };
    Vector<EarlyEvent> early;
    unsigned synthetic[2][NumTierExperts] { };   // superseded un-acted Go votes
    unsigned syntheticJsc[2][NumTierExperts] { };
};

// Call-edge state (Supply and tier-inversion), kept apart from FnState so a call link does not
// allocate a full FnState (2136 bytes) for caller and callee (D21).
struct EdgeState {
    CodeBlock* slots[kSupplySlots] { };
    bool deferred[kSupplySlots] { };
    unsigned nextSlot { 0 };
    bool pulled { false };
    HashSet<CodeBlock*> trackedBy; // callers whose slots hold this function
};

struct FnState { // per baseline CodeBlock
    // quiescence
    uint64_t fingerprint { 0 };
    bool haveFingerprint { false };
    MonotonicTime lastChange;
    bool requireChange { false };
    // queue-slack
    uint64_t lastJettisonTickPlusOne { 0 };
    MonotonicTime lastJettison; // lean Queue-Slack: recent-jettison window by timestamp
    // mixer: online jettison cost for synthetic pricing (sum, count) per step
    double onlineEarly[2][2] { };
    // mixer: losses not yet priceable (no speedup yet), applied when it is: raw ms per pair
    // ((B,D), (B,F), (D,F)) per expert, for (1 - 1/s) losses and (s - 1) losses; unpriced
    // synthetic votes per step per expert. Multiplicative updates commute, so deferral loses
    // only timeliness.
    double pendingFaster[2][3][NumTierExperts] { }; // [step][pair][expert]
    double pendingSlower[2][3][NumTierExperts] { };
    unsigned pendingSynthetic[2][NumTierExperts] { };
    bool hasPending { false };
    // ---- ghost scoring. Step si: 0 = Baseline->DFG, 1 = DFG->FTL. Tier index t: 0 Baseline, 1 DFG, 2 FTL.
    Vote latest[2][NumTierExperts] { };          // each tier-up expert's vote at the most recent decision point
    uint8_t latestAwake[2] { 0, 0 }; // experts awake (not abstaining) at that point
    bool haveLatest[2] { false, false };
    uint8_t compileYesMask[2] { 0, 0 };   // at the acted Go: experts that voted Go
    uint8_t compileAwakeMask[2] { 0, 0 }; // at the acted Go: experts awake
    bool ghostActive[2][NumTierExperts] { };     // voted Go while the acted decision was Wait
    uint64_t ghostFingerprint[2][NumTierExperts] { };
    uint64_t ghostPoint[2][NumTierExperts] { };
    uint8_t waiting[2] { 0, 0 };      // voted Wait at the acted Go and have not re-voted Go since
    uint8_t waitingAwake[2] { 0, 0 }; // experts awake at that acted Go
    // jettison cost accounting: lower-tier time (ms) by lower tier (0 Baseline, 1 DFG)
    bool installed[2] { false, false };
    double installedLowerMs[2][2] { };  // exits finishing in a lower tier while installed
    bool pendingEarly[2] { false, false };
    uint8_t pendingYes[2] { 0, 0 }, pendingAwake[2] { 0, 0 };
    double pendingLowerMs[2][2] { };    // the above, plus lower-tier time since the jettison
    RawLoss raw;
    // dfg-probe: the DFG code block seen at the previous DFG->FTL visit and its OSR exit count then
    CodeBlock* probeOwner { nullptr };
    uint32_t probeExits { 0 };
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
HashMap<CodeBlock*, std::unique_ptr<EdgeState>>* s_edges;
HashMap<CodeBlock*, std::unique_ptr<OptState>>* s_opt;
Stats s_stats;
Vector<Record>* s_retired;
// mixer: log-weights (sum of -lambda * loss / unit) per tier-up expert, at three levels:
// global, function class (0..3), and bucket (class, step). The bucket level alone is the D5
// mixer; --tierUpMixerPool=hierarchy decides on the sum of all three (D12).
double s_logGlobal[NumTierExperts] { };
double s_logClass[4][NumTierExperts] { };
double s_logBucket[4][2][NumTierExperts] { };
bool s_poolHierarchy { false };
bool s_weightedMajority { false };
unsigned s_incumbent { 0 }; // the expert whose vote stands on exact ties (D18; default jsc-default)
uint64_t s_mixerChose[NumTierExperts] { };
uint64_t s_weightUpdates { 0 }, s_unpricedUpdates { 0 }, s_majorityOverrodeJsc { 0 };
// queue-wait measurement per compile tier: plans, plans that arrived behind >= 1 / >= 2 plans,
// total wait ms, waits >= 1 ms, max wait ms
struct QueueStats { uint64_t plans { 0 }, behind1 { 0 }, behind2 { 0 }, waits1ms { 0 }; double waitMs { 0 }, maxWaitMs { 0 }; };
QueueStats s_queue[3];
Lock s_queueLock;
// online speedup aggregates (per class 0..3 + global 4, per pair): ms lo, work lo, ms hi, work hi
double s_onlineAgg[5][3][4] { };
MonotonicTime s_lastAggRefresh;
// online jettison costs for synthetic pricing, per class (0..3) and step
double s_onlineClassEarly[4][2][2] { }; // sum, count
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

EdgeState* findEdgeState(CodeBlock* baseline) WTF_REQUIRES_LOCK(s_lock)
{
    auto it = s_edges->find(baseline);
    return it == s_edges->end() ? nullptr : it->value.get();
}

EdgeState& edgeState(CodeBlock* baseline) WTF_REQUIRES_LOCK(s_lock)
{
    auto& p = s_edges->add(baseline, nullptr).iterator->value;
    if (!p)
        p = std::make_unique<EdgeState>();
    return *p;
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

std::optional<double> onlineSpeedup(CodeBlock*, FnState&, unsigned) WTF_REQUIRES_LOCK(s_lock);
void chargeFaster(CodeBlock*, FnState&, unsigned step, unsigned e, unsigned p, double ms) WTF_REQUIRES_LOCK(s_lock);
void chargeWeight(FnState&, unsigned step, unsigned e, double ms) WTF_REQUIRES_LOCK(s_lock);
bool mixerOn();

void closePendingEarly(FnState& st, unsigned step, CodeBlock* baseline = nullptr) WTF_REQUIRES_LOCK(s_lock)
{
    if (!st.pendingEarly[step])
        return;
    if (baseline && mixerOn()) {
        // Online price for the mixer, the same formula as the end-of-run pricing; parts whose
        // speedup is not known yet are deferred. The cost is also recorded (when fully priced)
        // to price synthetic votes.
        double cost = 0;
        bool priced = true;
        struct Part { double ms; unsigned pair; };
        Vector<Part, 2> parts;
        if (!step)
            parts.append({ st.pendingLowerMs[0][0], 0 });
        else {
            parts.append({ st.pendingLowerMs[1][0], 1 });
            parts.append({ st.pendingLowerMs[1][1], 2 });
        }
        for (auto& part : parts) {
            if (!(part.ms > 0))
                continue;
            auto sp = onlineSpeedup(baseline, st, part.pair);
            if (sp)
                cost += part.ms * (1 - 1 / *sp);
            else
                priced = false;
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if ((st.pendingYes[step] & (1 << e)) && (s_mix & (1 << e)))
                    chargeFaster(baseline, st, step, e, part.pair, part.ms);
            }
        }
        if (priced) {
            st.onlineEarly[step][0] += cost;
            st.onlineEarly[step][1]++;
            unsigned c = std::max<int8_t>(st.functionClass, 0);
            s_onlineClassEarly[c][step][0] += cost;
            s_onlineClassEarly[c][step][1]++;
        }
    }
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

// ---------------------------------------------------------------- mixer: online pricing
bool mixerOn() { return s_tierSelected == TMixer; }

// Online speedup for pair p ((B,D), (B,F), (D,F)): the function's own in-run data, else its
// class's aggregate, else the run's global aggregate (aggregates refreshed every 100 ms).
std::optional<double> onlineSpeedup(CodeBlock* baseline, FnState& st, unsigned p) WTF_REQUIRES_LOCK(s_lock)
{
    constexpr unsigned pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
    unsigned lo = pairs[p][0], hi = pairs[p][1];
    uint64_t wlo = workOf(baseline, st, lo), whi = workOf(baseline, st, hi);
    if (st.sampleMs[lo] > 0 && wlo && st.sampleMs[hi] > 0 && whi)
        return (st.sampleMs[lo] / wlo) / (st.sampleMs[hi] / whi);
    for (unsigned c : { static_cast<unsigned>(std::max<int8_t>(st.functionClass, 0)), 4u }) {
        const double* a = s_onlineAgg[c][p];
        if (a[0] > 0 && a[1] > 0 && a[2] > 0 && a[3] > 0)
            return (a[0] / a[1]) / (a[2] / a[3]);
    }
    return std::nullopt;
}

void refreshOnlineAggregates() WTF_REQUIRES_LOCK(s_lock)
{
    constexpr unsigned pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
    double agg[5][3][4] { };
    for (auto& [function, st] : *s_fn) {
        uint64_t w[3] = { workOf(function, *st, 0), workOf(function, *st, 1), workOf(function, *st, 2) };
        for (unsigned p = 0; p < 3; ++p) {
            unsigned lo = pairs[p][0], hi = pairs[p][1];
            if (!(st->sampleMs[lo] > 0 && w[lo] && st->sampleMs[hi] > 0 && w[hi]))
                continue;
            for (unsigned c : { static_cast<unsigned>(std::max<int8_t>(st->functionClass, 0)), 4u }) {
                agg[c][p][0] += st->sampleMs[lo];
                agg[c][p][1] += w[lo];
                agg[c][p][2] += st->sampleMs[hi];
                agg[c][p][3] += w[hi];
            }
        }
    }
    memcpy(s_onlineAgg, agg, sizeof(agg));
}

// Multiplicative-weights update of expert e in the function's (class, step) bucket for a loss of
// `ms` extra time; one unit = one sampling interval (the analogue of one cache miss).
void chargeWeight(FnState& st, unsigned step, unsigned e, double ms) WTF_REQUIRES_LOCK(s_lock)
{
    if (!mixerOn() || !(s_mix & (1 << e)) || !(ms > 0))
        return;
    unsigned c = std::max<int8_t>(st.functionClass, 0);
    double delta = -kLearningRate * ms / intervalMs();
    s_logBucket[c][step][e] += delta;
    s_logClass[c][e] += delta;
    s_logGlobal[e] += delta;
    s_weightUpdates++;
}

// Charge a loss of `ms` lower-tier time that pair p's higher tier would have run faster:
// ms x (1 - 1/s). Deferred if the speedup is not known yet.
void chargeFaster(CodeBlock* baseline, FnState& st, unsigned step, unsigned e, unsigned p, double ms) WTF_REQUIRES_LOCK(s_lock)
{
    if (auto sp = onlineSpeedup(baseline, st, p))
        chargeWeight(st, step, e, ms * (1 - 1 / *sp));
    else {
        st.pendingFaster[step][p][e] += ms;
        st.hasPending = true;
        s_unpricedUpdates++;
    }
}

// Charge `ms` of higher-tier time that the lower tier would have taken longer: ms x (s - 1).
void chargeSlower(CodeBlock* baseline, FnState& st, unsigned step, unsigned e, unsigned p, double ms) WTF_REQUIRES_LOCK(s_lock)
{
    if (auto sp = onlineSpeedup(baseline, st, p))
        chargeWeight(st, step, e, ms * (*sp - 1));
    else {
        st.pendingSlower[step][p][e] += ms;
        st.hasPending = true;
        s_unpricedUpdates++;
    }
}

std::optional<double> onlineEarlyMean(FnState& st, unsigned step) WTF_REQUIRES_LOCK(s_lock)
{
    if (st.onlineEarly[step][1])
        return st.onlineEarly[step][0] / st.onlineEarly[step][1];
    unsigned c = std::max<int8_t>(st.functionClass, 0);
    if (s_onlineClassEarly[c][step][1])
        return s_onlineClassEarly[c][step][0] / s_onlineClassEarly[c][step][1];
    return std::nullopt;
}

// Apply deferred losses whose speedup / jettison cost is now known (every aggregate refresh).
void applyPending() WTF_REQUIRES_LOCK(s_lock)
{
    for (auto& [function, st] : *s_fn) {
        if (!st->hasPending)
            continue;
        bool left = false;
        for (unsigned p = 0; p < 3; ++p) {
            auto sp = onlineSpeedup(function, *st, p);
            for (unsigned step = 0; step < 2; ++step) {
                for (unsigned e = 0; e < NumTierExperts; ++e) {
                    if (!st->pendingFaster[step][p][e] && !st->pendingSlower[step][p][e])
                        continue;
                    if (!sp) {
                        left = true;
                        continue;
                    }
                    chargeWeight(*st, step, e, st->pendingFaster[step][p][e] * (1 - 1 / *sp));
                    chargeWeight(*st, step, e, st->pendingSlower[step][p][e] * (*sp - 1));
                    st->pendingFaster[step][p][e] = st->pendingSlower[step][p][e] = 0;
                }
            }
        }
        for (unsigned step = 0; step < 2; ++step) {
            auto mean = onlineEarlyMean(*st, step);
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if (!st->pendingSynthetic[step][e])
                    continue;
                if (!mean) {
                    left = true;
                    continue;
                }
                chargeWeight(*st, step, e, st->pendingSynthetic[step][e] * *mean);
                st->pendingSynthetic[step][e] = 0;
            }
        }
        st->hasPending = left;
    }
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
    double classMs[6][3][2] { }, classWork[6][3][2] { }; // index 5: all functions (global level)
    auto cls = [](const Record& r) { return r.functionClass < 0 ? 4u : static_cast<unsigned>(r.functionClass); };
    for (auto& r : records) {
        for (unsigned p = 0; p < 3; ++p) {
            unsigned lo = pairs[p][0], hi = pairs[p][1];
            if (r.ms[lo] > 0 && r.work[lo] && r.ms[hi] > 0 && r.work[hi]) {
                for (unsigned c : { cls(r), 5u }) {
                    classMs[c][p][0] += r.ms[lo];
                    classWork[c][p][0] += r.work[lo];
                    classMs[c][p][1] += r.ms[hi];
                    classWork[c][p][1] += r.work[hi];
                }
            }
        }
    }
    unsigned fromFunction = 0, fromClass = 0, fromGlobal = 0, missing = 0;
    auto speedup = [&](const Record& r, unsigned p) -> std::optional<double> {
        unsigned lo = pairs[p][0], hi = pairs[p][1];
        if (r.ms[lo] > 0 && r.work[lo] && r.ms[hi] > 0 && r.work[hi]) {
            fromFunction++;
            return (r.ms[lo] / r.work[lo]) / (r.ms[hi] / r.work[hi]);
        }
        // Fallbacks, all from this run: the function's class, then every function.
        unsigned c = cls(r);
        if (classMs[c][p][0] > 0 && classMs[c][p][1] > 0) {
            fromClass++;
            return (classMs[c][p][0] / classWork[c][p][0]) / (classMs[c][p][1] / classWork[c][p][1]);
        }
        if (classMs[5][p][0] > 0 && classMs[5][p][1] > 0) {
            fromGlobal++;
            return (classMs[5][p][0] / classWork[5][p][0]) / (classMs[5][p][1] / classWork[5][p][1]);
        }
        missing++;
        return std::nullopt;
    };

    double late[NumTierExperts] { }, delayed[NumTierExperts] { }, early[NumTierExperts] { }, synthetic[NumTierExperts] { };
    double shadow[NumTierExperts] { };
    double unpricedMs = 0;
    double unpricedByExpert[NumTierExperts] { };
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
            } else {
                unpricedMs += r.raw.late[0][e];
                unpricedByExpert[e] += r.raw.late[0][e];
            }
            if (sDF) {
                late[e] += r.raw.late[1][e] * (1 - 1 / *sDF);
                shadow[e] += r.raw.lateJsc[1][e] * (1 - 1 / *sDF);
            } else {
                unpricedMs += r.raw.late[1][e];
                unpricedByExpert[e] += r.raw.late[1][e];
            }
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
                    } else {
                        unpricedMs += raw;
                        unpricedByExpert[e] += raw;
                    }
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
        " unpricedMs=", unpricedMs, " speedupFromFunction=", fromFunction, " fromClass=", fromClass, " fromGlobal=", fromGlobal, " missing=", missing,
        " reevaluations=", s_stats.reevaluations);
    for (unsigned e = 0; e < NumTierExperts; ++e) {
        double own = late[e] + delayed[e] + early[e] + synthetic[e];
        dataLog(" | ", kTierNames[e], " late/delayed/early/synthetic=", late[e], "/", delayed[e], "/", early[e], "/", synthetic[e],
            " jscAtAwake=", shadow[e], " regret=", own - shadow[e], " unpricedRawMs=", unpricedByExpert[e]);
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
        " mix=", static_cast<unsigned>(s_mix), " pulls=", s_stats.pulls, " quietGaps=", s_stats.quietGaps);
    {
        Locker queueLocker { s_queueLock };
        const char* tiers[] = { "Baseline", "DFG", "FTL" };
        for (unsigned t = 0; t < 3; ++t) {
            const QueueStats& q = s_queue[t];
            dataLogLn("tierUpExperts: queue tier=", tiers[t], " plans=", q.plans, " arrivedBehind>=1=", q.behind1, " >=2=", q.behind2,
                " waits>=1ms=", q.waits1ms, " totalWaitMs=", q.waitMs, " maxWaitMs=", q.maxWaitMs);
        }
    }
    if (mixerOn()) {
        dataLog("tierUpExperts: mixer rule=", s_weightedMajority ? "weighted-majority" : "argmax", " pool=", s_poolHierarchy ? "hierarchy" : "bucket",
            " weightUpdates=", s_weightUpdates, " unpricedUpdates=", s_unpricedUpdates, " majorityOverrodeJsc=", s_majorityOverrodeJsc, " chose");
        for (unsigned e = 0; e < NumTierExperts; ++e)
            dataLog(" ", kTierNames[e], "=", s_mixerChose[e]);
        dataLogLn();
        for (unsigned c = 0; c < 4; ++c) {
            for (unsigned st = 0; st < 2; ++st) {
                dataLog("tierUpExperts: mixer logWeights class=", c, " step=", st);
                for (unsigned e = 0; e < NumTierExperts; ++e) {
                    if (s_mix & (1 << e))
                        dataLog(" ", kTierNames[e], "=", s_logBucket[c][st][e]);
                }
                dataLogLn();
            }
        }
    }
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
    s_edges = new HashMap<CodeBlock*, std::unique_ptr<EdgeState>>();
    s_opt = new HashMap<CodeBlock*, std::unique_ptr<OptState>>();
    s_retired = new Vector<Record>();
    s_waiters = new HashSet<CodeBlock*>();
    s_gapLow = new std::priority_queue<double>();
    s_gapHigh = new std::priority_queue<double, std::vector<double>, std::greater<double>>();
    bool found = !t;
    for (unsigned e = 0; t && e <= TMixer; ++e) {
        if (is(t, kTierNames[e])) {
            s_tierSelected = e;
            found = true;
        }
    }
    RELEASE_ASSERT_WITH_MESSAGE(found, "tierUpExperts: unknown --tierUpExpert");
    // Lean build: --tierUpMixExperts lists the experts that compute votes (jsc-default always
    // does); unset = every expert (the full ghost machinery used for validation).
    if (const char8_t* mix = Options::tierUpMixExperts()) {
        s_mix = 1 << TJsc;
        for (auto name : String::fromUTF8(reinterpret_cast<const char*>(mix)).split(',')) {
            bool known = false;
            for (unsigned e = 0; e < NumTierExperts; ++e) {
                if (name == String::fromLatin1(kTierNames[e])) {
                    s_mix |= 1 << e;
                    known = true;
                }
            }
            RELEASE_ASSERT_WITH_MESSAGE(known, "tierUpExperts: unknown expert in --tierUpMixExperts");
        }
    }
    RELEASE_ASSERT_WITH_MESSAGE(s_tierSelected == TMixer || (s_mix & (1 << s_tierSelected)), "tierUpExperts: the acting expert must be in --tierUpMixExperts");
    auto option = [](const char8_t* o) { return o ? String::fromUTF8(reinterpret_cast<const char*>(o)) : String(); };
    String pool = option(Options::tierUpMixerPool());
    String rule = option(Options::tierUpMixerRule());
    RELEASE_ASSERT_WITH_MESSAGE(pool.isNull() || pool == "bucket"_s || pool == "hierarchy"_s, "tierUpExperts: --tierUpMixerPool=bucket|hierarchy");
    RELEASE_ASSERT_WITH_MESSAGE(rule.isNull() || rule == "argmax"_s || rule == "weighted-majority"_s, "tierUpExperts: --tierUpMixerRule=argmax|weighted-majority");
    s_poolHierarchy = pool == "hierarchy"_s;
    if (const char8_t* inc = Options::tierUpMixerIncumbent()) {
        bool known = false;
        for (unsigned e = 0; e < NumTierExperts; ++e) {
            if (!strcmp(reinterpret_cast<const char*>(inc), kTierNames[e])) {
                s_incumbent = e;
                known = true;
            }
        }
        RELEASE_ASSERT_WITH_MESSAGE(known, "tierUpExperts: unknown --tierUpMixerIncumbent");
    }
    s_weightedMajority = rule == "weighted-majority"_s;
    g_deoptEnabled = !!d;
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
    if (auto e = s_edges->find(codeBlock); e != s_edges->end()) {
        EdgeState& es = *e->value;
        for (CodeBlock* caller : es.trackedBy) {
            if (EdgeState* c = findEdgeState(caller)) {
                for (auto& slot : c->slots) {
                    if (slot == codeBlock)
                        slot = nullptr;
                }
            }
        }
        for (CodeBlock* callee : es.slots) {
            if (EdgeState* c = callee ? findEdgeState(callee) : nullptr)
                c->trackedBy.remove(codeBlock);
        }
        s_edges->remove(e);
    }
    auto it = s_fn->find(codeBlock);
    if (it != s_fn->end()) {
        FnState& st = *it->value;
        for (CodeBlock* inlinee : st.inlinees) {
            if (FnState* i = findFnState(inlinee))
                i->inlinedInto.remove(codeBlock);
        }
        for (CodeBlock* inliner : st.inlinedInto) {
            if (FnState* i = findFnState(inliner))
                i->inlinees.removeAll(codeBlock);
        }
        // Keep its scoring data for the end-of-run pricing, if it has any.
        bool hasData = st.haveLatest[0] || st.haveLatest[1] || st.pendingEarly[0] || st.pendingEarly[1]
            || !st.raw.early.isEmpty() || st.sampleMs[0] > 0 || st.sampleMs[1] > 0 || st.sampleMs[2] > 0;
        if (hasData)
            s_retired->append(makeRecord(codeBlock, st));
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
// tier-inversion (D20): Go iff a tracked inlinable callee (Supply's call-edge slots) already runs
// at or above the target tier (DFG at step 0, FTL at step 1); never Wait.
Vote tierInversionVote(EdgeState* es, bool ftl) WTF_REQUIRES_LOCK(s_lock)
{
    if (!es)
        return Abstain;
    unsigned budget = inlineBudget(ftl);
    for (unsigned i = 0; i < kSupplySlots; ++i) {
        CodeBlock* callee = es->slots[i];
        if (!callee || callee->bytecodeCost() > budget)
            continue;
        CodeBlock* replacement = callee->replacement();
        JITType t = replacement ? replacement->jitType() : callee->jitType();
        if (ftl ? t == JITType::FTLJIT : (t == JITType::DFGJIT || t == JITType::FTLJIT))
            return Yes;
    }
    return Abstain;
}

// dfg-probe (D20): at DFG->FTL, Go iff the DFG code's OSR exit count is unchanged since this
// function's previous DFG->FTL visit with the same DFG code; Wait if it rose; abstain otherwise.
Vote dfgProbeVote(FnState& st, CodeBlock* dfg)
{
    if (!dfg || dfg->jitType() != JITType::DFGJIT || st.probeOwner != dfg)
        return Abstain;
    return dfg->osrExitCounter() == st.probeExits ? Yes : No;
}

// The pure (state-preserving) vote of tier-up expert `e` on `function` at step `si`, used to
// re-evaluate experts that voted Wait at an acted Go. jsc-default's rule is a counter crossing,
// which cannot be re-asked once the function has compiled: a waiting jsc-default stays waiting.
Vote pureVote(unsigned e, unsigned si, CodeBlock* function, FnState& st, uint64_t fingerprint, bool heavy, JITType top,
    int64_t idle, uint64_t tick, MonotonicTime now) WTF_REQUIRES_LOCK(s_lock)
{
    switch (e) {
    case TSupply: {
        EdgeState* es = findEdgeState(function);
        if (!es)
            return Abstain;
        unsigned budget = inlineBudget(si == 1);
        bool tracked = false;
        for (unsigned i = 0; i < kSupplySlots; ++i) {
            CodeBlock* callee = es->slots[i];
            if (!callee || callee->bytecodeCost() > budget)
                continue;
            tracked = true;
            if (!es->deferred[i] && isThin(callee))
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
    case TDfgProbe:
        return si == 1 ? dfgProbeVote(st, function->replacement()) : Abstain;
    case TTierInversion:
        return tierInversionVote(findEdgeState(function), si == 1);
    case TQueueSlack: {
        UNUSED_PARAM(tick);
        bool recentJettison = st.lastJettison && (now - st.lastJettison).milliseconds() < 20;
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
        if (mixerOn() && (now - s_lastAggRefresh).milliseconds() >= 100) {
            s_lastAggRefresh = now;
            refreshOnlineAggregates();
            applyPending();
        }
        if ((now - s_lastReevaluation).milliseconds() < 1 || s_waiters->isEmpty())
            return;
        s_lastReevaluation = now;
        functions = copyToVector(*s_waiters);
    }
    int64_t idle[2] = { idleCompilerSlots(false), idleCompilerSlots(true) };
    uint64_t tick = TierUpTimeAttribution::currentTick();
    for (CodeBlock* function : functions) {
        uint64_t fingerprint = (s_mix & (1 << TQuiescence)) ? profileFingerprint(function) : 0;
        bool heavy = (s_mix & (1 << TSampleGravity)) && TierUpTimeAttribution::inLowerTierHeavySet(function);
        CodeBlock* replacement = function->replacement();
        JITType top = replacement ? replacement->jitType() : function->jitType();
        Locker locker { s_lock };
        FnState* st = findFnState(function);
        if (!st)
            continue;
        for (unsigned si = 0; si < 2; ++si) {
            for (unsigned e = 1; e < NumTierExperts; ++e) {
                if (!(st->waiting[si] & (1 << e)) || !(s_mix & (1 << e)))
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

// Ghost scoring of one decision point, after the decision, so the profile fingerprint is
// computed only when needed: to check an active ghost, or to snapshot an un-acted Go.
template<typename ScoringFingerprintOf>
void scoreDecision(CodeBlock* baseline, unsigned si, const Vote* v, Vote acted, std::optional<uint64_t> ownFingerprint, const ScoringFingerprintOf& scoringFingerprintOf)
{
    // --tierUpGhostNoSynthetic: no synthetic losses, so no ghost snapshots and no fingerprint.
    bool synthetic = !Options::tierUpGhostNoSynthetic();
    bool needFingerprint = false;
    if (synthetic) {
        Locker locker { s_lock };
        FnState& st = fnState(baseline);
        for (unsigned e = 0; e < NumTierExperts; ++e)
            needFingerprint |= st.ghostActive[si][e] || (acted == No && v[e] == Yes);
    }
    uint64_t scoringFp = needFingerprint ? scoringFingerprintOf(ownFingerprint ? *ownFingerprint : profileFingerprint(baseline)) : 0;

    Locker locker { s_lock };
    FnState& st = fnState(baseline);
    uint64_t point = ++s_pointId;
    classOf(baseline, st);

    // Un-acted Go votes whose profile has since changed: their compile would have exited.
    uint8_t superseded = 0;
    for (unsigned e = 0; e < NumTierExperts; ++e) {
        if (st.ghostActive[si][e] && st.ghostFingerprint[si][e] != scoringFp) {
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
        if (mixerOn() && (s_mix & (1 << e))) {
            // Online price: this function's mean jettison cost at this step so far, else its
            // class's; deferred until one has been observed.
            if (auto mean = onlineEarlyMean(st, si))
                chargeWeight(st, si, e, *mean);
            else {
                st.pendingSynthetic[si][e]++;
                st.hasPending = true;
                s_unpricedUpdates++;
            }
        }
    }

    uint8_t awake = 0;
    for (unsigned e = 0; e < NumTierExperts; ++e) {
        st.latest[si][e] = v[e];
        if (v[e] != Abstain)
            awake |= 1 << e;
    }
    st.latestAwake[si] = awake;
    st.haveLatest[si] = true;

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
        st.compileAwakeMask[si] = awake;
        // Experts that voted Wait are charged the extra time they would have kept the function
        // in the lower tier, until their re-evaluated vote turns Go.
        st.waiting[si] = wait;
        st.waitingAwake[si] = awake;
        if (wait)
            s_waiters->add(baseline);
        return;
    }
    for (unsigned e = 0; e < NumTierExperts && synthetic; ++e) {
        if (v[e] == Yes && !st.ghostActive[si][e]) {
            st.ghostActive[si][e] = true;
            st.ghostFingerprint[si][e] = scoringFp;
            st.ghostPoint[si][e] = point;
        }
    }
}

bool tierUp(DP dp, CodeBlock* owner, bool jscGo)
{
    reevaluateWaiters();
    CodeBlock* baseline = baselineOf(owner);
    unsigned dpi = static_cast<unsigned>(dp);
    bool ftl = isFTLDP(dp);
    bool lb = dp == DP::LLIntToBaseline;

    // Inputs gathered outside s_lock. Lean build: only the mix's experts compute their inputs.
    bool inMix[NumTierExperts];
    for (unsigned e = 0; e < NumTierExperts; ++e)
        inMix[e] = s_mix & (1 << e);
    // The profile fingerprint feeds Quiescence and ghost scoring's synthetic exits. Without
    // Quiescence in the mix it is computed only when scoring needs it (after the decision).
    std::optional<uint64_t> fingerprint;
    if (!lb && inMix[TQuiescence])
        fingerprint = profileFingerprint(baseline);
    // Ghost scoring: the sites a ghost compile would have used are approximated by this
    // function's own profiles plus those of the inlinees of its most recent real compile
    // (approved approximation (a), validated by the ghost-vs-live ranking check).
    auto scoringFingerprintOf = [&](uint64_t own) {
        Vector<CodeBlock*> inlinees;
        {
            Locker locker { s_lock };
            if (FnState* st = findFnState(baseline))
                inlinees = st->inlinees;
        }
        uint64_t h = own;
        for (CodeBlock* inlinee : inlinees)
            h = WTF::pairIntHash(static_cast<unsigned>(h), static_cast<unsigned>(profileFingerprint(inlinee))) ^ (h << 1);
        return h;
    };
    MonotonicTime now = MonotonicTime::now();
    bool heavy = !lb && inMix[TSampleGravity] && TierUpTimeAttribution::inLowerTierHeavySet(baseline) && owner->jitType() != JITType::FTLJIT;
    int64_t idle = (lb || !inMix[TQueueSlack]) ? 0 : idleCompilerSlots(ftl);
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
        EdgeState* es = findEdgeState(baseline); // unique_ptr values: stable across map growth
        if (!inMix[TSupply])
            v[TSupply] = Abstain;
        else if (lb)
            v[TSupply] = es && es->pulled ? Yes : Abstain;
        else if (!isLoopDP(dp) && !es)
            v[TSupply] = Abstain;
        else if (!isLoopDP(dp)) {
            unsigned budget = inlineBudget(ftl);
            bool tracked = false;
            for (unsigned i = 0; i < kSupplySlots && v[TSupply] != No; ++i) {
                CodeBlock* callee = es->slots[i];
                if (!callee || callee->bytecodeCost() > budget)
                    continue;
                tracked = true;
                if (!es->deferred[i] && isThin(callee)) {
                    es->deferred[i] = true;
                    edgeState(callee).pulled = true;
                    pullTarget = callee;
                    v[TSupply] = No;
                }
            }
            if (v[TSupply] != No)
                v[TSupply] = tracked ? Yes : Abstain;
        }

        // quiescence
        if (!lb && inMix[TQuiescence]) {
            bool changed = !st.haveFingerprint || st.fingerprint != *fingerprint;
            if (changed) {
                if (st.haveFingerprint) {
                    addGap((now - st.lastChange).milliseconds());
                    st.requireChange = false;
                }
                st.fingerprint = *fingerprint;
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
        if (!lb && inMix[TSampleGravity])
            v[TSampleGravity] = heavy ? Yes : No;

        // queue-slack
        if (!lb && inMix[TQueueSlack]) {
            // Recent jettison: within the last two 10 ms periods, by timestamp (no sampler needed).
            UNUSED_PARAM(tick);
            bool recentJettison = st.lastJettison && (now - st.lastJettison).milliseconds() < 20;
            if (idle > 0)
                v[TQueueSlack] = recentJettison ? Abstain : Yes;
            else {
                unsigned tier = ftl ? 1 : 0;
                bool small = s_haveEwma[tier] && cost <= s_ewma[tier];
                v[TQueueSlack] = (small || isLoopDP(dp)) ? Yes : No;
            }
        }

        // dfg-probe: DFG->FTL only (owner is the DFG code block)
        if (inMix[TDfgProbe]) {
            if (dp == DP::DFGToFTL) {
                v[TDfgProbe] = dfgProbeVote(st, owner);
                st.probeOwner = owner;
                st.probeExits = owner->osrExitCounter();
            } else
                v[TDfgProbe] = Abstain;
        }

        // tier-inversion: function-entry points only
        if (inMix[TTierInversion])
            v[TTierInversion] = (!lb && !isLoopDP(dp)) ? tierInversionVote(es, ftl) : Abstain;

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
        if (mixerOn()) {
            unsigned c = classOf(baseline, st);
            unsigned si = ftl ? 1 : 0;
            auto score = [&](unsigned e) {
                double l = s_logBucket[c][si][e];
                if (s_poolHierarchy)
                    l += s_logClass[c][e] + s_logGlobal[e];
                return l;
            };
            if (!s_weightedMajority) {
                // D5: the awake expert in the mix with the largest weight acts (ties: jsc-default,
                // then lower index).
                unsigned chosen = (inMix[s_incumbent] && v[s_incumbent] != Abstain) ? s_incumbent : TJsc;
                for (unsigned e = 0; e < NumTierExperts; ++e) {
                    if (inMix[e] && v[e] != Abstain && score(e) > score(chosen))
                        chosen = e;
                }
                s_mixerChose[chosen]++;
                acted = v[chosen];
            } else {
                // D12: weighted majority (Littlestone-Warmuth) of the awake experts' votes; an exact
                // tie keeps jsc-default's vote.
                double top = score(TJsc);
                for (unsigned e = 1; e < NumTierExperts; ++e) {
                    if (inMix[e] && v[e] != Abstain)
                        top = std::max(top, score(e));
                }
                double go = 0, wait = 0;
                for (unsigned e = 0; e < NumTierExperts; ++e) {
                    if (!inMix[e] || v[e] == Abstain)
                        continue;
                    (v[e] == Yes ? go : wait) += std::exp(score(e) - top);
                }
                // Exact tie: the incumbent's vote (jsc-default if the incumbent abstains).
                Vote tie = (inMix[s_incumbent] && v[s_incumbent] != Abstain) ? v[s_incumbent] : v[TJsc];
                acted = go > wait ? Yes : wait > go ? No : tie;
                if (acted != v[TJsc])
                    s_majorityOverrodeJsc++;
            }
        } else
            acted = v[s_tierSelected] == Abstain ? v[TJsc] : v[s_tierSelected];

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
            if (lb && es)
                es->pulled = false;
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

    if (!lb)
        scoreDecision(baseline, ftl ? 1 : 0, v, acted, fingerprint, scoringFingerprintOf);

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
    if (!(s_mix & ((1 << TSupply) | (1 << TTierInversion))))
        return; // call edges feed only Supply and tier-inversion
    CodeBlock* callerCodeBlock = dynamicDowncast<CodeBlock>(caller);
    if (!callerCodeBlock || callerCodeBlock == callee)
        return;
    if (callee->jitType() != JITType::InterpreterThunk || callee->codeType() != FunctionCode || callee->bytecodeCost() > inlineBudget(true))
        return;
    CodeBlock* callerBaseline = baselineOf(callerCodeBlock);
    Locker locker { s_lock };
    EdgeState& c = edgeState(callerBaseline);
    for (CodeBlock* slot : c.slots) {
        if (slot == callee)
            return;
    }
    unsigned i = c.nextSlot++ % kSupplySlots;
    if (CodeBlock* old = c.slots[i]) {
        if (EdgeState* o = findEdgeState(old))
            o->trackedBy.remove(callerBaseline);
    }
    c.slots[i] = callee;
    c.deferred[i] = false;
    edgeState(callee).trackedBy.add(callerBaseline);
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
        closePendingEarly(st, tier, baseline);
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
        st.lastJettison = MonotonicTime::now();
        if (st.probeOwner == optimized)
            st.probeOwner = nullptr; // dfg-probe: a later DFG block may reuse this address
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
                if (st->latest[si][e] == No) {
                    st->raw.late[si][e] += ms;
                    if (mixerOn() && (s_mix & (1 << e)))
                        chargeFaster(root, *st, si, e, si ? 2 : 0, ms);
                }
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
        // Kept in Baseline instead of DFG (B,D) / FTL (B,F), or in DFG instead of FTL (D,F).
        for (unsigned e = 0; e < NumTierExperts; ++e) {
            if (st->waiting[si] & (1 << e)) {
                st->raw.delayed[si][h][e] += ms;
                if (mixerOn() && (s_mix & (1 << e)))
                    chargeSlower(root, *st, si, e, si ? 2 : (h ? 1 : 0), ms);
            }
            if ((st->waiting[si] & (1 << TJsc)) && (st->waitingAwake[si] & (1 << e)))
                st->raw.delayedJsc[si][h][e] += ms;
        }
    }
}

void noteQueueWait(unsigned tier, double waitMs, unsigned ahead, size_t)
{
    if (tier >= 3)
        return;
    Locker locker { s_queueLock };
    QueueStats& q = s_queue[tier];
    q.plans++;
    q.behind1 += ahead >= 1;
    q.behind2 += ahead >= 2;
    q.waits1ms += waitMs >= 1;
    q.waitMs += waitMs;
    q.maxWaitMs = std::max(q.maxWaitMs, waitMs);
}

bool isWorkCounted(CodeBlock* codeBlock)
{
    ScriptExecutable* executable = codeBlock->ownerExecutable();
    unsigned position = (static_cast<unsigned>(executable->firstLine()) << 12) ^ static_cast<unsigned>(executable->startColumn());
    return !(WTF::intHash(position) % std::max(1u, Options::tierUpExpertsWorkCountedOneIn()));
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
