/*
 * minuteman diagnostic: see ProfileConsumptionLog.h.
 */

#include "config.h"
#include "ProfileConsumptionLog.h"

#include "CodeBlock.h"
#include "JSCInlines.h"
#include <cstdio>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/SixCharacterHash.h>
#include <wtf/Vector.h>
#include <wtf/text/StringBuilder.h>

namespace JSC {
namespace ProfileConsumptionLog {

namespace {

struct Block {
    String id;
    int tier { 0 };          // 0 LLInt, 1 Baseline, 2 other
    unsigned live { 0 }, total { 0 };
    StringBuilder statuses;
    unsigned nStatuses { 0 };
};

struct Compile {
    String machine;
    int mode { 0 };
    double tMs { 0 };
    Vector<std::pair<CodeBlock*, std::unique_ptr<Block>>> blocks;
};

thread_local Compile* t_compile { nullptr };
Lock s_fileLock;
FILE* s_file { nullptr };
MonotonicTime s_start;

String idFor(CodeBlock* codeBlock)
{
    if (codeBlock->hasHash()) {
        auto h = integerToSixCharacterHashString(codeBlock->hash().hash());
        StringBuilder sb;
        for (char c : h)
            sb.append(static_cast<Latin1Character>(c));
        return sb.toString();
    }
    return makeString(String::fromUTF8(codeBlock->inferredName().span()), "@"_s, codeBlock->ownerExecutable()->firstLine());
}

Block& blockFor(CodeBlock* profiled)
{
    for (auto& [cb, b] : t_compile->blocks) {
        if (cb == profiled)
            return *b;
    }
    auto b = std::make_unique<Block>();
    b->id = idFor(profiled);
    JITType type = profiled->jitType();
    b->tier = type == JITType::InterpreterThunk ? 0 : type == JITType::BaselineJIT ? 1 : 2;
    profiled->forEachValueProfile([&](auto& profile, bool) {
        b->total++;
        if (profile.m_prediction != SpecNone)
            b->live++;
    });
    Block& ref = *b;
    t_compile->blocks.append({ profiled, WTF::move(b) });
    return ref;
}

} // anonymous namespace

CompileScope::CompileScope(CodeBlock* machine, CodeBlock* profiledMachine, int mode)
{
    if (!isEnabled())
        return;
    {
        Locker locker { s_fileLock };
        if (!s_file) {
            s_file = fopen(reinterpret_cast<const char*>(Options::profileConsumptionLogFile()), "w");
            s_start = MonotonicTime::now();
        }
    }
    t_compile = new Compile;
    t_compile->machine = idFor(machine);
    t_compile->mode = mode;
    t_compile->tMs = (MonotonicTime::now() - s_start).milliseconds();
    blockFor(profiledMachine);
}

CompileScope::~CompileScope()
{
    if (!t_compile)
        return;
    StringBuilder line;
    line.append("{\"machine\":\""_s, t_compile->machine, "\",\"mode\":"_s, t_compile->mode, ",\"t_ms\":"_s, t_compile->tMs, ",\"blocks\":["_s);
    bool first = true;
    for (auto& [cb, b] : t_compile->blocks) {
        line.append(first ? ""_s : ","_s, "{\"id\":\""_s, b->id, "\",\"tier\":"_s, b->tier, ",\"live\":"_s, b->live,
            ",\"total\":"_s, b->total, ",\"statuses\":["_s, b->statuses.toString(), "]}"_s);
        first = false;
    }
    line.append("]}\n"_s);
    CString utf8 = line.toString().utf8();
    {
        Locker locker { s_fileLock };
        if (s_file) {
            fwrite(utf8.data(), 1, utf8.length(), s_file);
            fflush(s_file);
        }
    }
    delete t_compile;
    t_compile = nullptr;
}

void noteInlinee(CodeBlock* profiledInlinee)
{
    if (!t_compile)
        return;
    blockFor(profiledInlinee);
}

void noteStatus(CodeBlock* profiledBlock, BytecodeIndex index, Access access, bool fromBaselineIC, int state, unsigned variants)
{
    if (!t_compile)
        return;
    Block& b = blockFor(profiledBlock);
    b.statuses.append(b.nStatuses ? ","_s : ""_s, "["_s, index.offset(), ","_s, access == Access::Get ? "\"g\""_s : "\"p\""_s,
        ","_s, fromBaselineIC ? 1 : 0, ","_s, state, ","_s, variants, "]"_s);
    b.nStatuses++;
}

} // namespace ProfileConsumptionLog
} // namespace JSC
