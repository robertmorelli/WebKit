/*
 * minuteman prototype: per-call-site argument SpeculatedType masks. See the header.
 */

#include "config.h"
#include "MinutemanCallSiteProfile.h"

#if ENABLE(MINUTEMAN_CALLSITE_PROFILING)

#include "BytecodeStructs.h"
#include "CallFrame.h"
#include "CallFrameInlines.h"
#include "CodeBlock.h"
#include "FunctionExecutable.h"
#include "JSCJSValueInlines.h"
#include "JSFunction.h"
#include "SpeculatedType.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>

namespace JSC {
namespace MinutemanCallSiteProfile {

namespace {

constexpr unsigned maxArgs = 8; // including |this|

struct Key {
    unsigned callerHash;
    unsigned bytecodeOffset;
    uintptr_t calleeIdentity;   // FunctionExecutable* (or JSObject* for host / non-function callees)
    bool operator==(const Key&) const = default;
};

struct KeyHash {
    size_t operator()(const Key& k) const
    {
        uint64_t h = (static_cast<uint64_t>(k.callerHash) << 32) ^ k.bytecodeOffset;
        h ^= static_cast<uint64_t>(k.calleeIdentity) * 0x9E3779B97F4A7C15ull;
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

struct Site {
    uint64_t count { 0 };
    unsigned argc { 0 };
    std::array<SpeculatedType, maxArgs> masks { };
    std::string callerHash, callerName, calleeHash, calleeName, kind;
    bool calleeIsJSFunction { false };
};

struct State {
    Lock lock;
    std::unordered_map<Key, Site, KeyHash> sites;
    uint64_t observed { 0 };
};

State& state();

std::string hashString(CodeBlockHash h)
{
    CString s = makeString(h).utf8();
    std::string out(byteCast<char>(s.data()), s.length());
    return out;
}

std::string toStd(const String& s)
{
    CString c = s.utf8();
    return std::string(byteCast<char>(c.data()), c.length());
}

void appendJSONString(std::string& out, const std::string& s)
{
    out += '"';
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        if (static_cast<unsigned char>(c) < 0x20)
            continue;
        out += c;
    }
    out += '"';
}

// Coarse, DFG-relevant classes of a SpeculatedType.
std::string classes(SpeculatedType t)
{
    struct C { SpeculatedType bits; const char* name; };
    static const C table[] = {
        { SpecInt32Only, "Int32" },
        { SpecInt52Any, "Int52" },
        { SpecDoubleReal | SpecDoublePureNaN | SpecDoubleImpureNaN, "Double" },
        { SpecBigInt, "BigInt" },
        { SpecBoolean, "Boolean" },
        { SpecOther, "Other" },
        { SpecString, "String" },
        { SpecSymbol, "Symbol" },
        { SpecFinalObject, "FinalObject" },
        { SpecArray, "Array" },
        { SpecFunction, "Function" },
        { SpecTypedArrayView, "TypedArray" },
    };
    std::string out;
    SpeculatedType rest = t;
    for (auto& c : table) {
        if (t & c.bits) {
            if (!out.empty())
                out += '|';
            out += c.name;
            rest &= ~c.bits;
        }
    }
    if (rest & SpecObject) {
        out += out.empty() ? "" : "|";
        out += "OtherObject";
        rest &= ~SpecObject;
    }
    if (rest & SpecCell) {
        out += out.empty() ? "" : "|";
        out += "OtherCell";
        rest &= ~SpecCell;
    }
    if (rest & ~SpecEmpty) {
        out += out.empty() ? "" : "|";
        out += "Misc";
    }
    return out.empty() ? "Empty" : out;
}

void dumpAtExit()
{
    const char8_t* path = Options::minutemanCallSiteProfileFile();
    State& s = state();
    Locker locker { s.lock };
    std::string out = "{\"observed_calls\":" + std::to_string(s.observed) + ",\"sample_rate\":"
        + std::to_string(Options::minutemanCallSiteProfileSampleRate()) + ",\"sites\":[\n";
    bool first = true;
    for (auto& [key, site] : s.sites) {
        if (!first)
            out += ",\n";
        first = false;
        out += "{\"caller\":";
        appendJSONString(out, site.callerHash);
        out += ",\"caller_name\":";
        appendJSONString(out, site.callerName);
        out += ",\"bc\":" + std::to_string(key.bytecodeOffset) + ",\"callee\":";
        appendJSONString(out, site.calleeHash);
        out += ",\"callee_name\":";
        appendJSONString(out, site.calleeName);
        out += ",\"callee_identity\":" + std::to_string(key.calleeIdentity);
        out += ",\"js_function\":" + std::string(site.calleeIsJSFunction ? "true" : "false");
        out += ",\"kind\":";
        appendJSONString(out, site.kind);
        out += ",\"count\":" + std::to_string(site.count) + ",\"argc\":" + std::to_string(site.argc) + ",\"args\":[";
        for (unsigned i = 0; i < std::min(site.argc, maxArgs); ++i) {
            if (i)
                out += ',';
            out += "{\"mask\":" + std::to_string(site.masks[i]) + ",\"classes\":";
            appendJSONString(out, classes(site.masks[i]));
            out += '}';
        }
        out += "]}";
    }
    out += "\n]}\n";
    FILE* f = path ? fopen(byteCast<char>(path), "w") : stderr;
    if (!f)
        return;
    fwrite(out.data(), 1, out.size(), f);
    if (f != stderr)
        fclose(f);
}

State& state()
{
    static LazyNeverDestroyed<State> s;
    static std::once_flag once;
    std::call_once(once, [] {
        s.construct();
        std::atexit(dumpAtExit);
    });
    return s.get();
}

template<typename Op>
void recordOp(CallFrame* callFrame, CodeBlock* caller, const JSInstruction* instruction, CodeSpecializationKind kind)
{
    auto op = instruction->as<Op>();
    // r(), not uncheckedR(): the callee operand may be a constant register (constant pool).
    JSValue calleeValue = callFrame->r(op.m_callee).jsValue();
    unsigned argc = op.m_argc; // including |this|
    Register* calleeFrame = reinterpret_cast<Register*>(callFrame) - op.m_argv;

    uintptr_t identity = 0;
    FunctionExecutable* executable = nullptr;
    if (calleeValue.isCell()) {
        if (auto* function = dynamicDowncast<JSFunction>(calleeValue.asCell())) {
            if (!function->isHostOrBuiltinFunction())
                executable = function->jsExecutable();
        }
        identity = executable ? reinterpret_cast<uintptr_t>(executable) : reinterpret_cast<uintptr_t>(calleeValue.asCell());
    }

    State& s = state();
    Locker locker { s.lock };
    s.observed++;
    Key key { caller->hash().hash(), caller->bytecodeOffset(instruction), identity };
    auto [it, isNew] = s.sites.try_emplace(key);
    Site& site = it->second;
    if (isNew) {
        site.callerHash = hashString(caller->hash());
        site.callerName = toStd(String::fromUTF8(caller->inferredName().span()));
        site.kind = kind == CodeSpecializationKind::CodeForConstruct ? "construct" : "call";
        site.calleeIsJSFunction = !!executable;
        if (executable)
            site.calleeName = toStd(executable->ecmaName().string());
    }
    if (site.calleeHash.empty() && executable) {
        if (CodeBlock* calleeCodeBlock = executable->codeBlockFor(kind))
            site.calleeHash = hashString(calleeCodeBlock->hash());
    }
    site.count++;
    site.argc = std::max(site.argc, argc);
    for (unsigned i = 0; i < std::min(argc, maxArgs); ++i) {
        JSValue v = (calleeFrame + CallFrame::argumentOffsetIncludingThis(i))->jsValue();
        site.masks[i] |= v ? speculationFromValue(v) : SpecEmpty;
    }
}

} // anonymous namespace

void record(CallFrame* callFrame, CodeBlock* caller, const JSInstruction* instruction)
{
    if (!Options::minutemanCallSiteProfile())
        return;
    unsigned rate = Options::minutemanCallSiteProfileSampleRate();
    if (rate > 1) {
        static thread_local unsigned counter = 0;
        if (++counter % rate)
            return;
    }
    switch (instruction->opcodeID()) {
    case op_call:
        recordOp<OpCall>(callFrame, caller, instruction, CodeSpecializationKind::CodeForCall);
        break;
    case op_call_ignore_result:
        recordOp<OpCallIgnoreResult>(callFrame, caller, instruction, CodeSpecializationKind::CodeForCall);
        break;
    case op_construct:
        recordOp<OpConstruct>(callFrame, caller, instruction, CodeSpecializationKind::CodeForConstruct);
        break;
    default:
        break; // varargs / eval / tail calls are not recorded
    }
}

} // namespace MinutemanCallSiteProfile
} // namespace JSC

#else

namespace JSC {
namespace MinutemanCallSiteProfile {
void record(CallFrame*, CodeBlock*, const JSInstruction*) { }
} // namespace MinutemanCallSiteProfile
} // namespace JSC

#endif // ENABLE(MINUTEMAN_CALLSITE_PROFILING)
