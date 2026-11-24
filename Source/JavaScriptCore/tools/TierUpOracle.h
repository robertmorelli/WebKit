/*
 * minuteman: exact per-function tier-up timings, for hindsight-oracle experiments.
 *
 * Enabled only with --tierUpOracleFile=<path>. With the option unset (the default) every
 * hook is a single null check on an Options string and behaviour matches upstream.
 * Functions not listed in the file keep JSC's default behaviour.
 *
 * File format: one line per function, "//" comment lines and blank lines ignored:
 *
 *     <CodeBlockHash> B=<n|never|default> D=<n|never|default> F=<n|never|default> osr=<on|off>
 *
 * Fields may be omitted (= default; osr defaults to on). Counts n are in JSC's own
 * execution-counter units: LLInt->Baseline +10 per return and +1 per loop iteration;
 * Baseline->DFG and DFG->FTL +15 per entry/return and +1 per loop iteration.
 *
 * For a listed function with a numeric count:
 *   B=n  the LLInt counter is armed to exactly n when the CodeBlock is created, bypassing
 *        UnlinkedCodeBlock::thresholdForJIT (x4 / /2 for cached code) and memory scaling.
 *   D=n  every Baseline->DFG arming (warm-up, long warm-up, the OSR-exit ramp's re-arm, and
 *        re-arms after a jettison or invalidated compile) uses exactly n, counted from that
 *        arming, bypassing the size scaling factor, the 2^retry reoptimization backoff, the
 *        quick-tier-up factor and memory scaling; shouldOptimizeNowFromBaseline()'s profile
 *        fullness/liveness delay is skipped.
 *   all  for an exact count the counter's "crossed" test is exact (count >= n): JSC's
 *        early-crossing leniency (up to half a checkpoint chunk, 15000 units for FTL) and
 *        memory-pressure scaling are skipped (ExecutionCounter::checkIfThresholdCrossedAndSet).
 *   F=n  the DFG->FTL counter is armed to exactly n when DFG code is installed and after a
 *        failed/invalidated FTL attempt, bypassing size scaling, the quick-FTL factor and
 *        memory scaling.
 *   never  B=never: never Baseline-JIT (stays in LLInt; implies no DFG/FTL).
 *          D=never: never DFG (implies no FTL).  F=never: never FTL.
 *   osr=off  no OSR entry from Baseline loops into DFG and no FTL OSR entry (no
 *          FTLForOSREntry compiles, loops get CheckTierUpInLoop instead of
 *          CheckTierUpAndOSREnter); tier-up then takes effect at the next invocation.
 *
 * JSC's short "check again soon" re-arms used while a compile is pending or after code is
 * ready (optimizeSoon) are left unchanged; they poll and never start a compile on their own.
 * OSR-exit-driven reoptimization (jettison) is NOT suppressed: speculation failures are a
 * consequence of the code, not a timing decision. With --verboseTierUpOracle every trigger
 * of a listed function is logged with its requested and actual count, and every jettison.
 */

#pragma once

#include "Options.h"
#include "ProfilerJettisonReason.h"
#include <cstdint>

namespace JSC {

class CodeBlock;

namespace TierUpOracle {

enum class Tier : uint8_t { Baseline = 0, DFG = 1, FTL = 2 };

// Per-tier setting: Default keeps JSC's logic, Never forbids the tier, otherwise an exact count.
constexpr int64_t Default = -1;
constexpr int64_t Never = -2;

struct Entry {
    int64_t count[3] { Default, Default, Default };
    bool osr { true };
};

inline bool isEnabled() { return !!Options::tierUpOracleFile(); }

// nullptr if `codeBlock` (any tier; its baseline's hash is used) is not listed.
const Entry* entryFor(CodeBlock*);

inline int64_t countFor(CodeBlock* codeBlock, Tier tier)
{
    const Entry* e = entryFor(codeBlock);
    return e ? e->count[static_cast<unsigned>(tier)] : Default;
}

inline bool osrAllowed(CodeBlock* codeBlock)
{
    const Entry* e = entryFor(codeBlock);
    return !e || e->osr;
}

inline bool hasExactCount(CodeBlock* codeBlock, Tier tier)
{
    int64_t n = countFor(codeBlock, tier);
    return n != Default && n != Never;
}

int32_t clampCount(int64_t);

// Verbose logging (--verboseTierUpOracle).
void logTrigger(CodeBlock*, Tier, double actualCount);
void logJettison(CodeBlock*, Profiler::JettisonReason);

} // namespace TierUpOracle

} // namespace JSC
