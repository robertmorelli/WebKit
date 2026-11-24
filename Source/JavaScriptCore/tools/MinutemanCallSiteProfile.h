/*
 * minuteman prototype: per-call-site argument SpeculatedType masks (polyvariance study).
 *
 * Compiled in only when ENABLE_MINUTEMAN_CALLSITE_PROFILING is defined (a separate
 * profiling build); in normal builds this header declares nothing that is called and the
 * LLInt/Baseline hooks are compiled out. In the profiling build, recording also requires
 * --minutemanCallSiteProfile=1.
 *
 * For every executed op_call / op_call_ignore_result / op_construct in LLInt and Baseline
 * (or 1 in --minutemanCallSiteProfileSampleRate), keyed by (caller CodeBlock hash, bytecode
 * index, callee executable): a call count and, per argument (this + up to 7), the OR of
 * speculationFromValue(). Written as JSON to --minutemanCallSiteProfileFile at exit.
 * Run with --useDFGJIT=false so every call stays in LLInt/Baseline and is observed.
 */

#pragma once

#include "Instruction.h"

namespace JSC {

class CallFrame;
class CodeBlock;

namespace MinutemanCallSiteProfile {

void record(CallFrame* callerFrame, CodeBlock* caller, const JSInstruction*);

} // namespace MinutemanCallSiteProfile

} // namespace JSC
