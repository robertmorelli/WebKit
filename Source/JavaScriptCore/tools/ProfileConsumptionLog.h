/*
 * minuteman diagnostic: what each DFG/FTL compile consumed from its inlinees' profiles.
 *
 * Enabled only with --profileConsumptionLogFile=<path> (default unset: one null check per
 * hook). For every optimizing compile, one JSON line: the machine CodeBlock, the mode, and for
 * the machine block and every inlinee the tier it was in when the parser read it (LLInt or
 * Baseline), how many of its value profiles were live, and every get_by_id / put_by_id status
 * the parser computed from it: whether the information came from a Baseline inline cache or
 * from LLInt's cache, the resulting status state and its number of variants (a Simple status
 * with one variant is what lets the DFG emit a direct structure-checked load or store).
 */

#pragma once

#include "BytecodeIndex.h"
#include "Options.h"
#include <cstdint>

namespace JSC {

class CodeBlock;

namespace ProfileConsumptionLog {

inline bool isEnabled() { return !!Options::profileConsumptionLogFile(); }

enum class Access : uint8_t { Get, Put };

// RAII scope for one parse; compile threads only.
class CompileScope {
public:
    CompileScope(CodeBlock* machine, CodeBlock* profiledMachine, int mode);
    ~CompileScope();
};

void noteInlinee(CodeBlock* profiledInlinee);
void noteStatus(CodeBlock* profiledBlock, BytecodeIndex, Access, bool fromBaselineIC, int state, unsigned variants);

} // namespace ProfileConsumptionLog
} // namespace JSC
