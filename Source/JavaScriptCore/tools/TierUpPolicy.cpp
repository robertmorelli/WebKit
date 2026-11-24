/*
 * minuteman: online, pretraining-free tier-up policy. See TierUpPolicy.h.
 */

#include "config.h"
#include "TierUpPolicy.h"

#include "CodeBlock.h"
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/WeakRandom.h>
#include <wtf/text/StringView.h>

namespace JSC {
namespace TierUpPolicy {

namespace {

constexpr unsigned numberOfExperts = 3;
constexpr double experts[numberOfExperts] { 0.25, 1.0, 4.0 };
constexpr unsigned numberOfClasses = 3;      // bytecode size buckets
constexpr int32_t minimumThreshold[2] { 1, 2 }; // same floors as --jitPolicyScale=0

struct Decision {
    uint8_t expert { 1 };
    uint8_t cls { 0 };
    int32_t baseThreshold { 0 };            // threshold before the multiplier, at first arming
    MonotonicTime triggeredAt { MonotonicTime::nan() };
};

struct State {
    Lock lock;
    HashMap<uintptr_t, Decision> decisions WTF_GUARDED_BY_LOCK(lock);
    double loss[numberOfClasses][numberOfExperts] WTF_GUARDED_BY_LOCK(lock) { };
    WeakRandom random WTF_GUARDED_BY_LOCK(lock);
    // stats
    uint64_t decisionsMade WTF_GUARDED_BY_LOCK(lock) { 0 };
    uint64_t chosen[numberOfExperts] WTF_GUARDED_BY_LOCK(lock) { };
    uint64_t earlyLosses WTF_GUARDED_BY_LOCK(lock) { 0 };
    uint64_t lateLosses WTF_GUARDED_BY_LOCK(lock) { 0 };
    uint64_t jettisons WTF_GUARDED_BY_LOCK(lock) { 0 };
    bool noop { false };   // "noop": full bookkeeping, but every multiplier is 1 (overhead measurement)

    State()
        : random(Options::tierUpPolicySeed())
    {
    }
};

void dumpStatsAtExit();

State& state()
{
    static LazyNeverDestroyed<State> s;
    static std::once_flag once;
    std::call_once(once, [] {
        s.construct();
        StringView name = StringView::fromLatin1(byteCast<char>(Options::tierUpPolicy()));
        s->noop = name == "noop"_s;
        if (name != "hedge"_s && !s->noop)
            dataLogLn("tierUpPolicy: unknown policy '", Options::tierUpPolicy(), "', using hedge");
        if (Options::verboseTierUpPolicy())
            std::atexit(dumpStatsAtExit);
    });
    return s.get();
}

uintptr_t key(CodeBlock* baseline, Step step)
{
    return (reinterpret_cast<uintptr_t>(baseline) << 1) | static_cast<uintptr_t>(step);
}

unsigned classOf(CodeBlock* codeBlock)
{
    if (!Options::tierUpPolicyPerClass())
        return 0;
    unsigned size = codeBlock->instructionsSize();
    return size < 100 ? 0 : size < 1000 ? 1 : 2;
}

void charge(State& s, unsigned cls, const bool blamed[numberOfExperts]) WTF_REQUIRES_LOCK(s.lock)
{
    double gamma = Options::tierUpPolicyGamma();
    for (unsigned e = 0; e < numberOfExperts; ++e)
        s.loss[cls][e] = gamma * s.loss[cls][e] + (blamed[e] ? 1.0 : 0.0);
}

unsigned sampleExpert(State& s, unsigned cls) WTF_REQUIRES_LOCK(s.lock)
{
    double eta = Options::tierUpPolicyEta();
    double minLoss = s.loss[cls][0];
    for (unsigned e = 1; e < numberOfExperts; ++e)
        minLoss = std::min(minLoss, s.loss[cls][e]);
    double w[numberOfExperts];
    double total = 0;
    for (unsigned e = 0; e < numberOfExperts; ++e)
        total += (w[e] = std::exp(-eta * (s.loss[cls][e] - minLoss)));
    double r = s.random.get() * total;
    for (unsigned e = 0; e < numberOfExperts; ++e) {
        if (r < w[e])
            return e;
        r -= w[e];
    }
    return numberOfExperts - 1;
}

void dumpStatsAtExit()
{
    State& s = state();
    Locker locker { s.lock };
    dataLogLn("tierUpPolicy: decisions=", s.decisionsMade, " chosen[0.25,1,4]=[", s.chosen[0], ",", s.chosen[1], ",", s.chosen[2],
        "] earlyDeoptLosses=", s.earlyLosses, " lateLosses=", s.lateLosses, " speculationJettisons=", s.jettisons);
    for (unsigned c = 0; c < (Options::tierUpPolicyPerClass() ? numberOfClasses : 1); ++c)
        dataLogLn("tierUpPolicy: class ", c, " discountedLoss=[", s.loss[c][0], ",", s.loss[c][1], ",", s.loss[c][2], "]");
}

} // anonymous namespace

int32_t adjustThreshold(CodeBlock* baseline, Step step, int32_t threshold)
{
    State& s = state();
    double multiplier;
    {
        Locker locker { s.lock };
        auto result = s.decisions.add(key(baseline, step), Decision { });
        Decision& d = result.iterator->value;
        if (result.isNewEntry) {
            d.cls = classOf(baseline);
            d.expert = sampleExpert(s, d.cls);
            d.baseThreshold = threshold;
            s.decisionsMade++;
            s.chosen[d.expert]++;
        }
        multiplier = s.noop ? 1.0 : experts[d.expert];
    }
    double scaled = static_cast<double>(threshold) * multiplier;
    if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
        return std::numeric_limits<int32_t>::max();
    return std::max(static_cast<int32_t>(scaled), minimumThreshold[static_cast<unsigned>(step)]);
}

void noteTrigger(CodeBlock* baseline, Step step)
{
    State& s = state();
    Locker locker { s.lock };
    auto it = s.decisions.find(key(baseline, step));
    if (it == s.decisions.end())
        return;
    Decision& d = it->value;
    if (!std::isnan(d.triggeredAt.secondsSinceEpoch().value()))
        return; // already fired once for this decision (e.g. a retried compile)
    d.triggeredAt = MonotonicTime::now();
    if (d.expert > 0) {
        double wasted = static_cast<double>(d.baseThreshold) * (experts[d.expert] - experts[0]);
        if (wasted * baseline->instructionsSize() / 1e5 > Options::tierUpPolicyLateM()) {
            bool blamed[numberOfExperts];
            for (unsigned e = 0; e < numberOfExperts; ++e)
                blamed[e] = experts[e] >= experts[d.expert];
            charge(s, d.cls, blamed);
            s.lateLosses++;
        }
    }
}

void noteJettison(CodeBlock* optimized, Profiler::JettisonReason reason)
{
    Step step;
    if (optimized->jitType() == JITType::DFGJIT)
        step = Step::DFG;
    else if (optimized->jitType() == JITType::FTLJIT)
        step = Step::FTL;
    else
        return;
    CodeBlock* baseline = optimized->baselineAlternative();
    if (!baseline)
        return;
    State& s = state();
    Locker locker { s.lock };
    auto it = s.decisions.find(key(baseline, step));
    if (it != s.decisions.end()) {
        Decision d = it->value;
        if (Profiler::isSpeculationFailure(reason)) {
            s.jettisons++;
            double sinceTrigger = (MonotonicTime::now() - d.triggeredAt).milliseconds();
            if (!std::isnan(sinceTrigger) && sinceTrigger <= Options::tierUpPolicyEarlyDeoptMs()) {
                bool blamed[numberOfExperts];
                for (unsigned e = 0; e < numberOfExperts; ++e)
                    blamed[e] = experts[e] <= experts[d.expert];
                charge(s, d.cls, blamed);
                s.earlyLosses++;
            }
        }
        s.decisions.remove(it);
    }
    // Losing the DFG code also ends the pending FTL decision for this function.
    if (step == Step::DFG)
        s.decisions.remove(key(baseline, Step::FTL));
}

} // namespace TierUpPolicy
} // namespace JSC
