/*
 * minuteman: online, pretraining-free tier-up policy ("hedge").
 *
 * Enabled only with --tierUpPolicy=hedge (or =noop: identical bookkeeping, every multiplier 1,
 * used to measure the policy's overhead). With the option unset (the default) every
 * hook is a single null check on an Options string and behaviour matches upstream.
 *
 * Experts: multipliers {0.25, 1, 4} on JSC's Baseline->DFG and DFG->FTL thresholds.
 * One decision is sampled per (baseline CodeBlock, step) when that step's threshold is
 * first armed, and reused until the step's code is jettisoned (re-armings such as
 * optimizeSoon() keep the same multiplier). Loop-triggered OSR entry into DFG/FTL uses
 * the same execution counters, so it is covered by the same decision.
 *
 * Losses, using only signals JSC can observe cheaply:
 *   early deopt  a DFG/FTL CodeBlock is jettisoned for a speculation failure
 *                (Profiler::isSpeculationFailure) within --tierUpPolicyEarlyDeoptMs of its
 *                tier-up firing -> +1 to every expert at least as eager as the chosen one.
 *   late (proxy) when a tier-up fires with a non-minimal expert, the most eager expert
 *                would have fired after (m_min / m_chosen) of the counter units; if the
 *                wasted units x bytecode size / 1e5 exceed --tierUpPolicyLateM -> +1 to
 *                every expert at least as patient as the chosen one. JSC has no cheap
 *                per-function time attribution, so this proxy depends only on threshold
 *                and size: in effect a tunable bias toward eager tier-up.
 * Update: discounted Hedge, L_e <- gamma * L_e + loss_e, P(e) ~ exp(-eta * L_e).
 * State: one global learner, or one per function class (3 size x 2 loopiness buckets)
 * with --tierUpPolicyPerClass. Per decision: O(#experts) work plus one hash-table lookup.
 */

#pragma once

#include "Options.h"
#include "ProfilerJettisonReason.h"
#include <cstdint>

namespace JSC {

class CodeBlock;

namespace TierUpPolicy {

enum class Step : uint8_t { DFG = 0, FTL = 1 };

inline bool isEnabled() { return !!Options::tierUpPolicy(); }

// Called where a tier-up threshold is armed; `baseline` is the baseline CodeBlock.
int32_t adjustThreshold(CodeBlock* baseline, Step, int32_t threshold);

// Called when a DFG/FTL compile is requested for `baseline` (the threshold fired).
void noteTrigger(CodeBlock* baseline, Step);

// Called from CodeBlock::jettison for DFG/FTL CodeBlocks.
void noteJettison(CodeBlock* optimized, Profiler::JettisonReason);

} // namespace TierUpPolicy

} // namespace JSC
