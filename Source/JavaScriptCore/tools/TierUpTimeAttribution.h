/*
 * minuteman: cheap per-function time attribution for tier-up policies.
 *
 * Enabled only with --tierUpTimeAttribution=1 (default off; with it off nothing is started
 * and the VM hooks are a single option check at VM construction/destruction).
 *
 * A background thread wakes every --tierUpTimeAttributionIntervalUs (default 1000 us,
 * +/-20% jitter as in SamplingProfiler), suspends the JS thread, reads only the TOP frame
 * (no stack walk, no allocation while suspended), and charges one sample to
 * (root function, tier):
 *   root function = the top frame's CodeBlock followed back through alternative() to the
 *                   baseline CodeBlock (all pointers validated against the CodeBlockSet);
 *   tier          = LLInt (PC in LLInt), Baseline / DFG / FTL (PC in JIT memory: the
 *                   CodeBlock's JIT type; inlined code is charged to the machine CodeBlock),
 *                   or Native (PC in C++; charged to vm.topCallFrame's function, if any).
 * This is the signal a bandit needs to observe "this function spent long in a lower tier".
 * --tierUpTimeAttributionSelfSampling replaces the suspending sampler with self-sampling: a
 * per-thread CPU-time timer delivers SIGPROF to the JS thread, whose async-signal-safe handler
 * pushes the raw PC and the CodeBlock slots of fp and vm.topCallFrame into a lock-free ring that
 * the sampler thread drains and validates every interval (see TierUpTimeAttribution.cpp).
 * --verboseTierUpTimeAttribution prints a summary at exit (hashes only where JSC had
 * already computed them; the sampler never computes a hash itself).
 */

#pragma once

#include "Options.h"
#include <cstdint>

namespace JSC {

class CodeBlock;
class VM;

namespace TierUpTimeAttribution {

enum Tier : unsigned { LLInt = 0, Baseline = 1, DFG = 2, FTL = 3, Native = 4, NumberOfTiers = 5 };

inline bool isEnabled() { return Options::tierUpTimeAttribution(); }

void start(VM&);   // on the JS thread, from VM construction
void stop(VM&);    // from VM destruction; joins the sampler thread

// Samples charged so far to `root` (a baseline CodeBlock) in `tier`.
uint64_t samples(CodeBlock* root, Tier);
// All samples charged so far to `root`, every tier.
uint64_t totalSamples(CodeBlock* root);
// Sampler ticks so far (one per interval, whether or not the JS thread was in JS).
uint64_t currentTick();
// Space-Saving heavy hitters (k = kHeavyHitters, Metwally et al. 2005) over samples that landed
// in LLInt, Baseline or DFG: is `root` in the current heavy set?
constexpr unsigned kHeavyHitters = 16;
bool inLowerTierHeavySet(CodeBlock* root);

} // namespace TierUpTimeAttribution

} // namespace JSC
