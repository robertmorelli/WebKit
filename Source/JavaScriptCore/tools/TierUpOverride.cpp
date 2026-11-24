/*
 * minuteman: per-CodeBlock tier-up overrides. See TierUpOverride.h for the file format.
 */

#include "config.h"
#include "TierUpOverride.h"

#include "CodeBlock.h"
#include <mutex>
#include <stdio.h>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/SafeStrerror.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {
namespace TierUpOverride {

namespace {

constexpr unsigned numberOfSteps = 3;

struct Entry {
    double scale[numberOfSteps] { 1.0, 1.0, 1.0 };
    bool forbid[numberOfSteps] { false, false, false };
};

using Map = HashMap<String, Entry>;

// Same floors scaleJITPolicy() uses for --jitPolicyScale=0.
constexpr int32_t minimumThreshold[numberOfSteps] { 0, 1, 2 };

const char* stepName(Step step)
{
    switch (step) {
    case Step::Baseline: return "baseline";
    case Step::DFG: return "dfg";
    case Step::FTL: return "ftl";
    }
    return "?";
}

bool parseLine(const String& rawLine, Map& map)
{
    String line = rawLine.trim(isASCIIWhitespace<char16_t>);
    if (line.isEmpty() || line.startsWith("//"_s))
        return true;

    Vector<String> fields = line.simplifyWhiteSpace(isASCIIWhitespace<char16_t>).split(' ');
    if (fields.size() != 2)
        return false;

    const String& hash = fields[0];
    String action = fields[1];
    std::optional<unsigned> onlyStep;
    if (size_t at = action.find('@'); at != notFound) {
        String tier = action.substring(at + 1);
        action = action.left(at);
        if (tier == "baseline"_s)
            onlyStep = static_cast<unsigned>(Step::Baseline);
        else if (tier == "dfg"_s)
            onlyStep = static_cast<unsigned>(Step::DFG);
        else if (tier == "ftl"_s)
            onlyStep = static_cast<unsigned>(Step::FTL);
        else
            return false;
    }

    auto applyScale = [&](Entry& entry, double factor) {
        for (unsigned i = 0; i < numberOfSteps; ++i) {
            if (!onlyStep || *onlyStep == i)
                entry.scale[i] *= factor;
        }
    };

    if (action == "default"_s) {
        map.add(hash, Entry { });
        return true;
    }
    if (action == "immediate"_s) {
        applyScale(map.add(hash, Entry { }).iterator->value, 0.0);
        return true;
    }
    if (action.startsWith("delay:"_s)) {
        bool ok = false;
        double factor = action.substring(6).toDouble(&ok);
        if (!ok || !(factor > 0))
            return false;
        applyScale(map.add(hash, Entry { }).iterator->value, factor);
        return true;
    }
    if (onlyStep)
        return false; // @tier only makes sense for immediate/delay.
    if (action == "neverDFG"_s) {
        auto& entry = map.add(hash, Entry { }).iterator->value;
        entry.forbid[static_cast<unsigned>(Step::DFG)] = true;
        entry.forbid[static_cast<unsigned>(Step::FTL)] = true;
        return true;
    }
    if (action == "neverFTL"_s) {
        map.add(hash, Entry { }).iterator->value.forbid[static_cast<unsigned>(Step::FTL)] = true;
        return true;
    }
    return false;
}

const Map& overrideMap()
{
    static LazyNeverDestroyed<Map> map;
    static std::once_flag once;
    std::call_once(once, [] {
        map.construct();
        const char8_t* filename = Options::tierUpOverrideFile();
        FILE* f = fopen(byteCast<char>(filename), "r");
        if (!f) {
            dataLogLn("tierUpOverrideFile: cannot open ", filename, ": ", safeStrerror(errno));
            return;
        }
        char buffer[BUFSIZ];
        unsigned lineNumber = 0;
        while (char* line = fgets(buffer, sizeof(buffer), f)) {
            ++lineNumber;
            String text = String::fromLatin1(line);
            if (!parseLine(text, map.get()))
                dataLogLn("tierUpOverrideFile: ", filename, ":", lineNumber, ": ignoring malformed line: ", text.trim(isASCIIWhitespace<char16_t>));
        }
        fclose(f);
        dataLogLnIf(Options::verboseOSR(), "tierUpOverrideFile: loaded ", map->size(), " entries from ", filename);
    });
    return map.get();
}

const Entry* entryFor(CodeBlock* codeBlock)
{
    const Map& map = overrideMap();
    if (map.isEmpty())
        return nullptr;
    auto it = map.find(makeString(codeBlock->hash()));
    return it == map.end() ? nullptr : &it->value;
}

} // anonymous namespace

int32_t adjustThreshold(CodeBlock* codeBlock, Step step, int32_t threshold)
{
    const Entry* entry = entryFor(codeBlock);
    if (!entry)
        return threshold;
    unsigned i = static_cast<unsigned>(step);
    double scaled = static_cast<double>(threshold) * entry->scale[i];
    int32_t result;
    if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
        result = std::numeric_limits<int32_t>::max();
    else
        result = std::max(static_cast<int32_t>(scaled), minimumThreshold[i]);
    dataLogLnIf(Options::verboseOSR(), "tierUpOverride: ", codeBlock->hash(), " ", stepName(step), " threshold ", threshold, " -> ", result);
    return result;
}

bool allowsCompile(CodeBlock* codeBlock, Step step)
{
    const Entry* entry = entryFor(codeBlock);
    return !entry || !entry->forbid[static_cast<unsigned>(step)];
}

} // namespace TierUpOverride
} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
