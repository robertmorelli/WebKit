/*
 * minuteman: structurally different tier-up and deopt experts for JSC, with ghost votes.
 *
 * Options (all default unset = upstream behaviour; every hook is one load of a cached bool):
 *   --tierUpExpert=jsc-default|supply|quiescence|sample-gravity|queue-slack
 *   --deoptExpert=jsc-default|refutation|cause-split
 *   --tierUpExpert=mixer: the full-information mixer (notes/decisions.md D5): the awake expert in
 *   --tierUpMixExperts with the largest multiplicative weight in the decision's (class, step)
 *   bucket acts; weights are charged online with the ghost losses.
 *   --tierUpMixExperts=<list>: lean build, only these experts (and jsc-default) compute votes.
 * Setting either option (jsc-default included) turns on the ghost machinery: at EVERY decision
 * point below, EVERY expert of that family computes its full vote (or abstains, sleeping-experts
 * style) and the vote is counted; only the selected expert's vote is acted on, and where it
 * abstains jsc-default acts. The experts read the 10 ms time attribution, so either option
 * requires --tierUpTimeAttribution=1 --tierUpTimeAttributionIntervalUs=10000.
 * --verboseTierUpExperts prints vote/abstain/act counts per expert per decision point at exit.
 *
 * Decision points (tier-up family):
 *   LLIntToBaseline  jitCompileAndSetHeuristics, every LLInt counter slow-path visit
 *   BaselineToDFG    operationOptimize, every Baseline counter visit at function entry
 *   BaselineLoopOSR  operationOptimize, every Baseline counter visit from a loop
 *   DFGToFTL         shouldTriggerFTLCompile from triggerTierUpNow
 *   DFGLoopOSR       shouldTriggerFTLCompile from tierUpCommon (loop tier-up / OSR entry)
 *   Retier           BaselineToDFG/BaselineLoopOSR visits of a function whose code a Cause-Split
 *                    jettison removed (deopt family; Cause-Split votes, others abstain)
 * Decision points (deopt family):
 *   Exit             every OSR exit whose kind may jettison (handleExitCounts calls in always)
 *   LoopReoptimize   operationOptimize's reoptimize-from-loop check
 *
 * Tier-up experts:
 *   jsc-default     JSC's own decision at the point (counter crossed and its profile gate).
 *   supply          (Caller-Pull + Supply's FTL budget.) Each caller tracks up to 4 linked callees
 *                   within the inline budget that were in LLInt when linked. At the caller's
 *                   DFG (FTL: larger budget) check: if a tracked callee is thin (still LLInt, or
 *                   Baseline whose inline caches were never considered) and that slot has not
 *                   deferred yet, pull the callee and Wait (once per slot); otherwise Go. Abstains
 *                   with nothing tracked and at OSR-entry points. At a pulled callee's
 *                   LLInt->Baseline check: Go.
 *   quiescence      64-bit fingerprint of value predictions, array modes, inline-cache states
 *                   (megamorphic caches excluded as terminal). A run-wide histogram of quiet gaps
 *                   (wall ms between fingerprint changes) that were later broken; Go iff the
 *                   current quiet gap exceeds its median. Abstains until the first broken gap.
 *                   After a jettison it requires one change, then quiet.
 *   sample-gravity  Space-Saving heavy hitters (k = 16) over 10 ms samples in LLInt/Baseline/
 *                   DFG; Go iff the function is in the heavy set and not yet FTL, else Wait.
 *                   Abstains at LLInt->Baseline.
 *   queue-slack     A free compile slot for the target tier (JITWorklist::idleCapacity: max
 *                   concurrent compiles - running - queued) and no jettison of this function in this or the
 *                   previous 10 ms tick: Go. No free slot: Go iff bytecode cost <= the EWMA
 *                   (weight 1/8) of completed compiles' cost for that tier, or this is a loop
 *                   OSR-entry point; else Wait. Abstains at LLInt->Baseline.
 * Deopt experts:
 *   jsc-default     JSC's own rule (exit count vs threshold, hot-baseline and stuck-in-loop).
 *   refutation      Jettison iff a recompile would compile differently at this exit: its
 *                   (bytecode, exit kind) is not yet a frequent exit site, or the value the exit
 *                   wrote back is outside the prediction the compile consumed at that profile.
 *                   Otherwise Keep, however many exits. At LoopReoptimize: Jettison iff an exit
 *                   of this code has refuted it.
 *   cause-split     Persistent kinds (BadType, BadConstantValue, BadExecutable, BadCache,
 *                   BadConstantCache, BadIndexingType): Jettison on the first exit at a site not
 *                   yet frequent, then re-arm "soon" and vote Go at the first Retier check where
 *                   that site's profile shows the new value. Transient kinds (Overflow,
 *                   NegativeZero, OutOfBounds, Uncountable, InadequateCoverage): Keep, unless the
 *                   same code origin exits in two distinct 10 ms ticks. Other kinds: abstain.
 */

#pragma once

#include "BytecodeIndex.h"
#include "CodeOrigin.h"
#include "ExitKind.h"
#include "MethodOfGettingAValueProfile.h"
#include "Options.h"
#include "ProfilerJettisonReason.h"
#include "SpeculatedType.h"
#include <cstdint>

namespace JSC {

class CodeBlock;
class JSCell;
class VM;

namespace DFG {
struct OSRExitBase;
}

namespace TierUpExperts {

enum class DP : uint8_t { LLIntToBaseline, BaselineToDFG, BaselineLoopOSR, DFGToFTL, DFGLoopOSR, Retier, Exit, LoopReoptimize, Count };

extern bool g_enabled;
extern bool g_deoptEnabled; // a --deoptExpert is set: exit ramps call in, refutation records predictions

void initialize(VM&);
void shutdown(VM&);
inline bool enabled() { return g_enabled; }
inline bool deoptEnabled() { return g_deoptEnabled; }

// ---- tier-up decision points. `jscGo` is JSC's own decision at this point, computed by the
// caller with JSC's code. Returns whether to compile now. If the acted decision is to wait while
// JSC would have gone, the expert re-arms JSC's "check again soon" itself.
bool tierUp(DP, CodeBlock* owner, bool jscGo);

// ---- deopt decision points.
// What an exit ramp needs at run time. DFG exits are decoded into temporaries when their ramp
// is compiled, so the ramp bakes a pointer to this copy, owned by the optimized CodeBlock's
// expert state and freed with it.
struct ExitSiteInfo {
    ExitKind kind;
    CodeOrigin codeOrigin;
    CodeOrigin codeOriginForExitProfile;
    MethodOfGettingAValueProfile valueProfile;
};
const ExitSiteInfo* retainExitSiteInfo(CodeBlock* optimized, const DFG::OSRExitBase&, const MethodOfGettingAValueProfile&);
// Exit ramp: records the exit and returns whether to jettison now.
bool exitDecision(CodeBlock* optimized, const ExitSiteInfo*);
// operationTriggerReoptimizationNow: consumes the request made by exitDecision.
bool takeJettisonRequest(CodeBlock* optimized);
// operationOptimize's reoptimize-from-loop check.
bool loopReoptimize(CodeBlock* optimized, bool jscJettison);

// ---- observations
void noteIncomingCall(CodeBlock* callee, JSCell* caller);
void noteOptimizedInstalled(CodeBlock* machine);
void noteJettison(CodeBlock* optimized, Profiler::JettisonReason); // after JSC re-armed the baseline
enum class ProfileKind : uint8_t { Bytecode, Argument, Lazy };
void noteConsumedPrediction(CodeBlock* machine, CodeBlock* profiled, ProfileKind, BytecodeIndex, uint64_t operandBits, SpeculatedType);
void forget(CodeBlock*);
// Work counter (calls + loop back-edges) for optimized code being compiled (compiler thread).
uint64_t* workCounterFor(CodeBlock* machine, bool ftl);
// From the time-attribution sampler thread: one sample charged to (`root`, tier), with tier as
// TierUpTimeAttribution::Tier. Drives the "too late" ghost loss.
void noteSample(CodeBlock* root, unsigned tier);

} // namespace TierUpExperts
} // namespace JSC
