/*
 * minuteman: per-CodeBlock tier-up overrides, for hindsight-oracle experiments.
 *
 * Enabled only when --tierUpOverrideFile=<path> is given. With the option unset
 * (the default) every hook below is a single null-check on an Options string and
 * the engine behaves exactly like upstream.
 *
 * File format: one entry per line, "//" starts a comment line, blank lines ignored.
 *
 *     <CodeBlockHash> <action>[@<tier>]
 *
 *   CodeBlockHash  6-character hash as printed by JSC (e.g. "CglQUp"), the same
 *                  hash used by --dfgAllowlist / the sampling profiler.
 *   action         default        no change (useful as an explicit no-op)
 *                  immediate      scale tier-up thresholds by 0 (clamped to the same
 *                                 minimums --jitPolicyScale=0 uses)
 *                  delay:<k>      scale tier-up thresholds by k (k > 0, real)
 *                  neverDFG       never DFG-compile this CodeBlock (implies no FTL)
 *                  neverFTL       never FTL-compile this CodeBlock
 *   @tier          optional, restricts immediate/delay to one tier-up step:
 *                  @baseline (LLInt->Baseline), @dfg (Baseline->DFG), @ftl (DFG->FTL).
 *                  Without it, immediate/delay apply to all three steps.
 *
 * Several lines may name the same hash; their effects compose (scales multiply).
 */

#pragma once

#include "Options.h"
#include <cstdint>

namespace JSC {

class CodeBlock;

namespace TierUpOverride {

enum class Step : uint8_t { Baseline = 0, DFG = 1, FTL = 2 };

inline bool isEnabled() { return !!Options::tierUpOverrideFile(); }

// Returns the threshold to use for this CodeBlock's tier-up step. Callers pass the
// threshold they would otherwise use; it is returned unchanged when no override applies.
int32_t adjustThreshold(CodeBlock*, Step, int32_t threshold);

// False if the override file forbids compiling this CodeBlock in this tier.
bool allowsCompile(CodeBlock*, Step);

} // namespace TierUpOverride

} // namespace JSC
