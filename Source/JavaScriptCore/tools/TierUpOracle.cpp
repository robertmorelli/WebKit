/*
 * minuteman: exact per-function tier-up timings. See TierUpOracle.h.
 */

#include "config.h"
#include "TierUpOracle.h"

#include "CodeBlock.h"
#include <array>
#include <mutex>
#include <stdio.h>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/SafeStrerror.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/StringToIntegerConversion.h>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {
namespace TierUpOracle {

namespace {

// Keyed by the numeric CodeBlockHash: lookups on hot slow paths allocate nothing.
using Map = HashMap<unsigned, Entry>;

bool parseValue(const String& v, int64_t& out)
{
    if (v == "default"_s) {
        out = Default;
        return true;
    }
    if (v == "never"_s) {
        out = Never;
        return true;
    }
    auto n = parseInteger<int64_t>(StringView { v });
    if (!n || *n < 1)
        return false;
    out = *n;
    return true;
}

bool parseLine(const String& raw, Map& map)
{
    String line = raw.trim(isASCIIWhitespace<char16_t>);
    if (line.isEmpty() || line.startsWith("//"_s))
        return true;
    Vector<String> fields = line.simplifyWhiteSpace(isASCIIWhitespace<char16_t>).split(' ');
    if (fields.size() < 2)
        return false;
    Entry e;
    for (size_t i = 1; i < fields.size(); ++i) {
        size_t eq = fields[i].find('=');
        if (eq == notFound)
            return false;
        String k = fields[i].left(eq);
        String v = fields[i].substring(eq + 1);
        if (k == "B"_s) {
            if (!parseValue(v, e.count[0]))
                return false;
        } else if (k == "D"_s) {
            if (!parseValue(v, e.count[1]))
                return false;
        } else if (k == "F"_s) {
            if (!parseValue(v, e.count[2]))
                return false;
        } else if (k == "osr"_s) {
            if (v == "on"_s)
                e.osr = true;
            else if (v == "off"_s)
                e.osr = false;
            else
                return false;
        } else
            return false;
    }
    if (fields[0].length() != CodeBlockHash::stringLength || !fields[0].containsOnlyASCII())
        return false;
    std::array<char, CodeBlockHash::stringLength> key;
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = static_cast<char>(fields[0][i]);
    unsigned h = CodeBlockHash(std::span<const char, CodeBlockHash::stringLength>(key)).hash();
    if (!h || h == std::numeric_limits<unsigned>::max())
        return false;
    map.set(h, e);
    return true;
}

const Map& oracleMap()
{
    static LazyNeverDestroyed<Map> map;
    static std::once_flag once;
    std::call_once(once, [] {
        map.construct();
        const char8_t* filename = Options::tierUpOracleFile();
        FILE* f = fopen(byteCast<char>(filename), "r");
        if (!f) {
            dataLogLn("tierUpOracleFile: cannot open ", filename, ": ", safeStrerror(errno));
            return;
        }
        char buffer[BUFSIZ];
        unsigned lineNumber = 0;
        while (char* line = fgets(buffer, sizeof(buffer), f)) {
            ++lineNumber;
            String text = String::fromLatin1(line);
            if (!parseLine(text, map.get()))
                dataLogLn("tierUpOracleFile: ", filename, ":", lineNumber, ": ignoring malformed line: ", text.trim(isASCIIWhitespace<char16_t>));
        }
        fclose(f);
        dataLogLnIf(Options::verboseTierUpOracle(), "tierUpOracle: loaded ", map->size(), " entries from ", filename);
    });
    return map.get();
}

const char* tierName(Tier t)
{
    switch (t) {
    case Tier::Baseline: return "B";
    case Tier::DFG: return "D";
    case Tier::FTL: return "F";
    }
    return "?";
}

} // anonymous namespace

const Entry* entryFor(CodeBlock* codeBlock)
{
    const Map& map = oracleMap();
    if (map.isEmpty() || !codeBlock)
        return nullptr;
    CodeBlock* baseline = codeBlock->baselineAlternative();
    unsigned h = (baseline ? baseline : codeBlock)->hash().hash();
    if (!h || h == std::numeric_limits<unsigned>::max())
        return nullptr;
    auto it = map.find(h);
    return it == map.end() ? nullptr : &it->value;
}

int32_t clampCount(int64_t n)
{
    if (n > std::numeric_limits<int32_t>::max())
        return std::numeric_limits<int32_t>::max();
    return static_cast<int32_t>(std::max<int64_t>(n, 1));
}

void logTrigger(CodeBlock* codeBlock, Tier tier, double actualCount)
{
    if (!Options::verboseTierUpOracle())
        return;
    const Entry* e = entryFor(codeBlock);
    if (!e)
        return;
    int64_t want = e->count[static_cast<unsigned>(tier)];
    dataLogLn("tierUpOracle: trigger ", codeBlock->hash(), " ", tierName(tier), " requested=",
        want == Default ? "default"_s : want == Never ? "never"_s : String::number(want), " actual=", actualCount);
}

void logJettison(CodeBlock* codeBlock, Profiler::JettisonReason reason)
{
    if (!Options::verboseTierUpOracle() || !entryFor(codeBlock))
        return;
    dataLogLn("tierUpOracle: jettison ", codeBlock->hash(), " ", codeBlock->jitType(), " reason=", reason);
}

} // namespace TierUpOracle
} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
