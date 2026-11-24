/*
 * minuteman: cheap per-function time attribution. See TierUpTimeAttribution.h.
 */

#include "config.h"
#include "TierUpTimeAttribution.h"

#include "CodeBlock.h"
#include "CodeBlockSet.h"
#include "ExecutableAllocator.h"
#include "JSCInlines.h"
#include "LLIntPCRanges.h"
#include "MachineContext.h"
#include "VM.h"
#include "TierUpExperts.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/SixCharacterHash.h>
#include <wtf/Threading.h>
#include <wtf/WeakRandom.h>

#if OS(LINUX) && CPU(X86_64)
#include <cerrno>
#include <signal.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
#define MINUTEMAN_SELF_SAMPLING 1
#else
#define MINUTEMAN_SELF_SAMPLING 0
#endif

namespace JSC {
namespace TierUpTimeAttribution {

namespace {

struct Entry {
    uint64_t counts[NumberOfTiers] { };
    unsigned hash { 0 };
};

#if MINUTEMAN_SELF_SAMPLING
// ---------------------------------------------------------------- self-sampling (signal mode)
// A per-thread CPU-time timer (timer_create(CLOCK_THREAD_CPUTIME_ID), SIGEV_THREAD_ID) delivers
// SIGPROF to the JS thread itself. The handler does only async-signal-safe work: it reads the
// interrupted PC and frame pointer from the ucontext, and, for frame pointers inside the JS
// thread's own stack, the CodeBlock slot of that frame and of vm.topCallFrame (plain stack
// loads). It pushes the raw words into a single-producer/single-consumer lock-free ring. The
// sampler thread drains the ring every interval and does all validation (CodeBlockSet, JIT
// memory ranges) under the same locks the suspend sampler uses. No thread is ever suspended.
//
// Differences from the suspend sampler, by construction: samples are spaced in the JS thread's
// CPU time, not wall time (a blocked JS thread is not sampled); a CodeBlock freed and its address
// reused between the signal and the drain (at most one interval) would be misattributed, so
// drained candidates are only counted if still in the CodeBlockSet.
// When the PC is in native code (a C++ operation or a thunk called from JS), vm.topCallFrame is
// not reliable: JIT code does not always publish it before calling out. So the handler also walks
// the saved frame-pointer chain (bounded, every load inside the JS stack) and records, for each
// frame, its return PC and the CodeBlock slot of the frame it returns into. The drain picks the
// first frame whose return PC is in JIT or LLInt code: the frame it returns into is the JS frame
// that made the call. topCallFrame is used only if no such frame is found.
constexpr unsigned kWalkDepth = 8;
struct RawSample {
    void* pc;
    CodeBlock* frameCodeBlock;          // fp's CodeBlock slot, if fp is inside the JS stack
    CodeBlock* topCodeBlock;            // vm.topCallFrame's CodeBlock slot (last resort)
    void* returnPC[kWalkDepth];         // frame i's saved return PC
    CodeBlock* callerCodeBlock[kWalkDepth]; // CodeBlock slot of frame i's caller
    uint8_t depth;
    bool inJS;                          // vm.entryScope != null when the signal arrived
};

constexpr unsigned kRingSize = 1024; // power of two; 10 ms samples drained every 10 ms
RawSample g_ring[kRingSize];
std::atomic<uint32_t> g_ringHead { 0 }; // written only by the signal handler (JS thread)
std::atomic<uint32_t> g_ringTail { 0 }; // written only by the drainer
std::atomic<uint64_t> g_signals { 0 }, g_dropped { 0 };
VM* g_signalVM { nullptr };
uint8_t* g_stackLow { nullptr };  // JS thread stack: [low, high)
uint8_t* g_stackHigh { nullptr };
timer_t g_timer;
bool g_timerArmed { false };
struct sigaction g_previousAction;

ALWAYS_INLINE bool inJSStack(const void* p)
{
    auto* q = static_cast<const uint8_t*>(p);
    return q >= g_stackLow && q + sizeof(Register) * CallFrame::headerSizeInRegisters <= g_stackHigh
        && !(reinterpret_cast<uintptr_t>(p) & (sizeof(void*) - 1));
}

SUPPRESS_ASAN void profHandler(int, siginfo_t*, void* context)
{
    int savedErrno = errno;
    g_signals.fetch_add(1, std::memory_order_relaxed);
    auto* uc = static_cast<ucontext_t*>(context);
    RawSample sample;
    sample.pc = reinterpret_cast<void*>(uc->uc_mcontext.gregs[REG_RIP]);
    void* fp = reinterpret_cast<void*>(uc->uc_mcontext.gregs[REG_RBP]);
    VM* vm = g_signalVM;
    sample.inJS = vm && vm->entryScope;
    sample.frameCodeBlock = inJSStack(fp) ? static_cast<CallFrame*>(fp)->unsafeCodeBlock() : nullptr;
    sample.depth = 0;
    // Saved-frame-pointer chain: [fp] = caller's fp, [fp + 8] = return PC (x86_64 SysV).
    void** frame = static_cast<void**>(fp);
    while (sample.depth < kWalkDepth && inJSStack(frame)) {
        void** caller = static_cast<void**>(frame[0]);
        if (!inJSStack(caller) || caller <= frame)
            break; // the stack grows down: a caller frame is always at a higher address
        sample.returnPC[sample.depth] = frame[1];
        sample.callerCodeBlock[sample.depth] = reinterpret_cast<CallFrame*>(caller)->unsafeCodeBlock();
        sample.depth++;
        frame = caller;
    }
    CallFrame* top = vm ? vm->topCallFrame : nullptr;
    sample.topCodeBlock = (top && inJSStack(top)) ? top->unsafeCodeBlock() : nullptr;
    uint32_t head = g_ringHead.load(std::memory_order_relaxed);
    if (head - g_ringTail.load(std::memory_order_acquire) >= kRingSize)
        g_dropped.fetch_add(1, std::memory_order_relaxed);
    else {
        g_ring[head & (kRingSize - 1)] = sample;
        g_ringHead.store(head + 1, std::memory_order_release);
    }
    errno = savedErrno;
}

// On the JS thread.
bool armSelfSampling(VM& vm, Seconds interval)
{
    g_signalVM = &vm;
    const StackBounds& stack = Thread::currentSingleton().stack();
    g_stackLow = static_cast<uint8_t*>(stack.end());     // stacks grow down: end() is the low bound
    g_stackHigh = static_cast<uint8_t*>(stack.origin());
    struct sigaction action { };
    action.sa_sigaction = profHandler;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGPROF, &action, &g_previousAction))
        return false;
    struct sigevent event { };
    event.sigev_notify = SIGEV_THREAD_ID;
    event.sigev_signo = SIGPROF;
    event._sigev_un._tid = static_cast<pid_t>(syscall(SYS_gettid));
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &event, &g_timer))
        return false;
    struct itimerspec spec { };
    int64_t ns = static_cast<int64_t>(interval.nanoseconds());
    spec.it_interval.tv_sec = ns / 1000000000;
    spec.it_interval.tv_nsec = ns % 1000000000;
    spec.it_value = spec.it_interval;
    if (timer_settime(g_timer, 0, &spec, nullptr)) {
        timer_delete(g_timer);
        return false;
    }
    g_timerArmed = true;
    return true;
}

void disarmSelfSampling()
{
    if (!g_timerArmed)
        return;
    timer_delete(g_timer);
    g_timerArmed = false;
    sigaction(SIGPROF, &g_previousAction, nullptr);
}
#endif // MINUTEMAN_SELF_SAMPLING

class Sampler {
public:
    Sampler(VM& vm)
        : m_vm(vm)
        , m_jsThread(Thread::currentSingleton())
        , m_interval(Seconds::fromMicroseconds(std::max(50u, Options::tierUpTimeAttributionIntervalUs())))
        , m_selfSampling(Options::tierUpTimeAttributionSelfSampling())
    {
    }

    bool selfSampling() const { return m_selfSampling; }
    Seconds interval() const { return m_interval; }

    void startThread()
    {
        m_thread = Thread::create("minuteman tier-up time attribution"_s, [this] { loop(); });
    }

    void stopThread()
    {
        m_shutdown.store(true);
        if (m_thread)
            m_thread->waitForCompletion();
        m_thread = nullptr;
    }

    uint64_t samples(CodeBlock* root, Tier tier)
    {
        Locker locker { m_dataLock };
        auto it = m_table.find(root);
        return it == m_table.end() ? 0 : it->value.counts[tier];
    }

    // Expert-facing inputs stay on a fixed 10 ms clock whatever the sampling interval, so a finer
    // interval (for ghost scoring) does not change what the experts decide: the tick counts 10 ms
    // periods, and the heavy-hitter set is fed one sample per 10 ms of samples.
    static constexpr unsigned kExpertClockUs = 10000;
    unsigned expertDecimation() const { return std::max(1u, kExpertClockUs / std::max(1u, static_cast<unsigned>(m_interval.microseconds()))); }
    uint64_t currentTick() const { return m_ticks.load(std::memory_order_relaxed) / expertDecimation(); }

    bool inLowerTierHeavySet(CodeBlock* root)
    {
        Locker locker { m_dataLock };
        for (unsigned i = 0; i < m_heavyCount; ++i) {
            if (m_heavy[i].key == root)
                return true;
        }
        return false;
    }

    uint64_t totalSamples(CodeBlock* root)
    {
        Locker locker { m_dataLock };
        auto it = m_table.find(root);
        if (it == m_table.end())
            return 0;
        uint64_t total = 0;
        for (unsigned t = 0; t < NumberOfTiers; ++t)
            total += it->value.counts[t];
        return total;
    }

    void dump()
    {
        Locker locker { m_dataLock };
        Vector<std::pair<uint64_t, Entry>> rows;
        uint64_t perTier[NumberOfTiers] { };
        for (auto& [cb, e] : m_table) {
            uint64_t total = 0;
            for (unsigned t = 0; t < NumberOfTiers; ++t) {
                total += e.counts[t];
                perTier[t] += e.counts[t];
            }
            rows.append({ total, e });
        }
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
        dataLogLn("tierUpTimeAttribution: ticks=", m_ticks.load(), " sampled=", m_sampled, " outsideJS=", m_outsideJS,
            " unattributed=", m_unattributed, " suspendFailed=", m_suspendFailed, " functions=", m_table.size(),
            " LLInt/Baseline/DFG/FTL/Native=", perTier[0], "/", perTier[1], "/", perTier[2], "/", perTier[3], "/", perTier[4],
            " meanSampleCostUs=", m_sampled ? m_sampleCostUs / m_sampled : 0.0,
            " meanLockWaitUs=", m_sampled ? m_lockWaitUs / m_sampled : 0.0,
            " meanSuspendedUs=", m_sampled ? m_suspendedUs / m_sampled : 0.0,
            " mode=", m_selfSampling ? "signal" : "suspend"
#if MINUTEMAN_SELF_SAMPLING
            , " signals=", g_signals.load(), " dropped=", g_dropped.load(),
            " nativeViaFrameWalk=", m_walkAttributed, " nativeViaTopCallFrame=", m_topCallFrameFallback
#endif
            );
        for (size_t i = 0; i < std::min<size_t>(rows.size(), 25); ++i) {
            auto& e = rows[i].second;
            auto h = e.hash ? integerToSixCharacterHashString(e.hash) : std::array<char, 6> { '?', '?', '?', '?', '?', '?' };
            dataLogLn("tierUpTimeAttribution: ", std::span<const char>(h), " total=", rows[i].first,
                " LLInt=", e.counts[0], " Baseline=", e.counts[1], " DFG=", e.counts[2], " FTL=", e.counts[3], " Native=", e.counts[4]);
        }
    }

private:
    bool isValidFramePointer(void* fp, const AbstractLocker& machineThreadsLocker)
    {
        uint8_t* p = static_cast<uint8_t*>(fp);
        for (auto& thread : m_vm.heap.machineThreads().threads(machineThreadsLocker)) {
            uint8_t* base = static_cast<uint8_t*>(thread->stack().origin());
            uint8_t* limit = static_cast<uint8_t*>(thread->stack().end());
            if (p < base && p >= limit)
                return true;
        }
        return false;
    }

    SUPPRESS_ASAN void takeSample()
    {
        m_ticks++;
        if (!m_vm.entryScope) {
            m_outsideJS++;
            return;
        }
        auto t0 = MonotonicTime::now();
        CodeBlock* root = nullptr;
        unsigned hash = 0;
        Tier tier = Native;
        {
            Locker machineThreadsLocker { m_vm.heap.machineThreads().getLock() };
            Locker codeBlockSetLocker { m_vm.heap.codeBlockSet().getLock() };
            std::optional<Locker<Lock>> executableAllocatorLocker;
            if (Options::useJIT())
                executableAllocatorLocker.emplace(ExecutableAllocator::singleton().getLock());
            auto tLocked = MonotonicTime::now();
            m_lockWaitUs += (tLocked - t0).microseconds();
            ThreadSuspendLocker threadSuspendLocker;
            if (!m_jsThread->suspend(threadSuspendLocker)) {
                m_suspendFailed++;
                return;
            }
            // No allocation while the JS thread is suspended.
            PlatformRegisters registers;
            m_jsThread->getRegisters(threadSuspendLocker, registers);
            void* fp = MachineContext::framePointer(registers);
            auto ip = MachineContext::instructionPointer(registers);
            void* pc = ip ? ip->untaggedPtr() : nullptr;
            CallFrame* frame = nullptr;
            enum { Jit, Interp, Other } kind = Other;
            if (Options::useJIT() && ExecutableAllocator::singleton().isValidExecutableMemory(*executableAllocatorLocker, pc)) {
                frame = static_cast<CallFrame*>(fp);
                kind = Jit;
            } else if (LLInt::isLLIntPC(pc)) {
                frame = static_cast<CallFrame*>(fp);
                kind = Interp;
            } else
                frame = m_vm.topCallFrame;
            CodeBlock* codeBlock = nullptr;
            if (frame && isValidFramePointer(frame, machineThreadsLocker)) {
                codeBlock = frame->unsafeCodeBlock();
                if (!m_vm.heap.codeBlockSet().contains(codeBlockSetLocker, codeBlock))
                    codeBlock = nullptr;
            }
            if (codeBlock) {
                if (kind == Interp)
                    tier = LLInt;
                else if (kind == Jit) {
                    switch (codeBlock->jitType()) {
                    case JITType::BaselineJIT: tier = Baseline; break;
                    case JITType::DFGJIT: tier = DFG; break;
                    case JITType::FTLJIT: tier = FTL; break;
                    default: tier = LLInt; break;
                    }
                } else
                    tier = Native;
                root = rootOf(codeBlock, codeBlockSetLocker, hash);
            }
            m_jsThread->resume(threadSuspendLocker);
            m_suspendedUs += (MonotonicTime::now() - tLocked).microseconds();
        }
        m_sampleCostUs += (MonotonicTime::now() - t0).microseconds();
        if (!root) {
            m_unattributed++;
            return;
        }
        record(root, tier, hash);
    }

    // Follows a validated CodeBlock back to its baseline root. Requires the CodeBlockSet lock.
    CodeBlock* rootOf(CodeBlock* codeBlock, const AbstractLocker& codeBlockSetLocker, unsigned& hash)
    {
        CodeBlock* root = codeBlock;
        for (unsigned i = 0; i < 8; ++i) {
            CodeBlock* next = root->alternative();
            if (!next || !m_vm.heap.codeBlockSet().contains(codeBlockSetLocker, next))
                break;
            root = next;
        }
        if (root->hasHash())
            hash = root->hash().hash();
        return root;
    }

    void record(CodeBlock* root, Tier tier, unsigned hash)
    {
        m_sampled++;
        {
            Locker locker { m_dataLock };
            Entry& e = m_table.add(root, Entry { }).iterator->value;
            e.counts[tier]++;
            if ((tier == LLInt || tier == Baseline || tier == DFG) && !(m_lowerTierSamples++ % expertDecimation()))
                updateHeavyHitters(root);
            if (hash)
                e.hash = hash;
        }
#if ENABLE(DFG_JIT)
        if (TierUpExperts::enabled())
            TierUpExperts::noteSample(root, tier);
#endif
    }

#if MINUTEMAN_SELF_SAMPLING
    // Signal mode: classify and record everything the handler pushed since the last drain.
    void drain()
    {
        m_ticks++;
        uint32_t tail = g_ringTail.load(std::memory_order_relaxed);
        uint32_t head = g_ringHead.load(std::memory_order_acquire);
        if (tail == head)
            return;
        auto t0 = MonotonicTime::now();
        Vector<std::tuple<CodeBlock*, Tier, unsigned>, 16> resolved;
        {
            Locker codeBlockSetLocker { m_vm.heap.codeBlockSet().getLock() };
            std::optional<Locker<Lock>> executableAllocatorLocker;
            if (Options::useJIT())
                executableAllocatorLocker.emplace(ExecutableAllocator::singleton().getLock());
            for (; tail != head; ++tail) {
                const RawSample& sample = g_ring[tail & (kRingSize - 1)];
                if (!sample.inJS) {
                    m_outsideJS++;
                    continue;
                }
                CodeBlock* candidate = nullptr;
                Tier tier = Native;
                if (LLInt::isLLIntPC(sample.pc)) {
                    candidate = sample.frameCodeBlock;
                    tier = LLInt;
                } else if (Options::useJIT() && ExecutableAllocator::singleton().isValidExecutableMemory(*executableAllocatorLocker, sample.pc)) {
                    candidate = sample.frameCodeBlock;
                    tier = FTL; // refined from the CodeBlock below
                } else {
                    // Native PC: the JS frame that called out, found by the frame-pointer walk;
                    // topCallFrame only if the walk found no frame returning into JS code.
                    for (unsigned i = 0; i < sample.depth && !candidate; ++i) {
                        void* returnPC = sample.returnPC[i];
                        if (LLInt::isLLIntPC(returnPC)
                            || (Options::useJIT() && ExecutableAllocator::singleton().isValidExecutableMemory(*executableAllocatorLocker, returnPC))) {
                            candidate = sample.callerCodeBlock[i];
                            m_walkAttributed++;
                        }
                    }
                    if (!candidate) {
                        candidate = sample.topCodeBlock;
                        m_topCallFrameFallback++;
                    }
                }
                if (!candidate || !m_vm.heap.codeBlockSet().contains(codeBlockSetLocker, candidate)) {
                    m_unattributed++;
                    continue;
                }
                if (tier == FTL) {
                    switch (candidate->jitType()) {
                    case JITType::BaselineJIT: tier = Baseline; break;
                    case JITType::DFGJIT: tier = DFG; break;
                    case JITType::FTLJIT: tier = FTL; break;
                    default: tier = LLInt; break;
                    }
                }
                unsigned hash = 0;
                CodeBlock* root = rootOf(candidate, codeBlockSetLocker, hash);
                resolved.append({ root, tier, hash });
            }
        }
        g_ringTail.store(tail, std::memory_order_release);
        for (auto& [root, tier, hash] : resolved)
            record(root, tier, hash);
        m_sampleCostUs += (MonotonicTime::now() - t0).microseconds();
    }
#endif

    // Space-Saving: increment if tracked; else fill a free slot; else replace the minimum
    // entry, inheriting its count + 1.
    void updateHeavyHitters(CodeBlock* root) WTF_REQUIRES_LOCK(m_dataLock)
    {
        unsigned minIndex = 0;
        for (unsigned i = 0; i < m_heavyCount; ++i) {
            if (m_heavy[i].key == root) {
                m_heavy[i].count++;
                return;
            }
            if (m_heavy[i].count < m_heavy[minIndex].count)
                minIndex = i;
        }
        if (m_heavyCount < kHeavyHitters) {
            m_heavy[m_heavyCount++] = { root, 1 };
            return;
        }
        m_heavy[minIndex] = { root, m_heavy[minIndex].count + 1 };
    }

    void loop()
    {
        while (!m_shutdown.load()) {
#if MINUTEMAN_SELF_SAMPLING
            if (m_selfSampling) {
                drain();
                WTF::sleep(m_interval);
                continue;
            }
#endif
            takeSample();
            double jitter = (m_random.get() * 2.0 - 1.0) * 0.2;
            WTF::sleep(m_interval * (1.0 + jitter));
        }
#if MINUTEMAN_SELF_SAMPLING
        if (m_selfSampling)
            drain();
#endif
    }

    VM& m_vm;
    Ref<Thread> m_jsThread;
    RefPtr<Thread> m_thread;
    Seconds m_interval;
    bool m_selfSampling;
    WeakRandom m_random { 0x51ed };
    std::atomic<bool> m_shutdown { false };
    Lock m_dataLock;
    HashMap<CodeBlock*, Entry> m_table WTF_GUARDED_BY_LOCK(m_dataLock);
    std::atomic<uint64_t> m_ticks { 0 };
    struct Heavy { CodeBlock* key; uint64_t count; };
    Heavy m_heavy[kHeavyHitters] WTF_GUARDED_BY_LOCK(m_dataLock) { };
    unsigned m_heavyCount WTF_GUARDED_BY_LOCK(m_dataLock) { 0 };
    uint64_t m_lowerTierSamples WTF_GUARDED_BY_LOCK(m_dataLock) { 0 };
    uint64_t m_walkAttributed { 0 }, m_topCallFrameFallback { 0 };
    uint64_t m_sampled { 0 }, m_outsideJS { 0 }, m_unattributed { 0 }, m_suspendFailed { 0 };
    double m_sampleCostUs { 0 };
    double m_lockWaitUs { 0 };    // waiting for heap/JIT locks before suspending (JS thread keeps running)
    double m_suspendedUs { 0 };   // suspend .. resume: the JS thread is stopped
};

Sampler* s_sampler { nullptr };
VM* s_vm { nullptr };
std::atomic<bool> s_dumped { false };

void dumpOnce()
{
    if (s_sampler && Options::verboseTierUpTimeAttribution() && !s_dumped.exchange(true))
        s_sampler->dump();
}

} // anonymous namespace

void start(VM& vm)
{
    if (s_sampler)
        return; // one VM (the jsc shell's main VM)
    s_vm = &vm;
    s_sampler = new Sampler(vm);
    if (s_sampler->selfSampling()) {
#if MINUTEMAN_SELF_SAMPLING
        RELEASE_ASSERT_WITH_MESSAGE(armSelfSampling(vm, s_sampler->interval()), "tierUpTimeAttribution: could not arm the SIGPROF thread CPU timer");
#else
        RELEASE_ASSERT_NOT_REACHED_WITH_MESSAGE("tierUpTimeAttributionSelfSampling is only implemented on Linux x86_64");
#endif
    }
    s_sampler->startThread();
    if (Options::verboseTierUpTimeAttribution())
        std::atexit(dumpOnce); // the jsc shell may exit without destroying its VM
}

void stop(VM& vm)
{
    if (!s_sampler || s_vm != &vm)
        return;
#if MINUTEMAN_SELF_SAMPLING
    if (s_sampler->selfSampling()) {
        disarmSelfSampling(); // on the JS thread: no SIGPROF after this
        g_signalVM = nullptr;
    }
#endif
    s_sampler->stopThread();
    dumpOnce();
    // Intentionally leaked: policies may still hold the pointer-keyed results until exit.
    s_vm = nullptr;
}

uint64_t samples(CodeBlock* root, Tier tier)
{
    return s_sampler ? s_sampler->samples(root, tier) : 0;
}

uint64_t currentTick()
{
    return s_sampler ? s_sampler->currentTick() : 0;
}

bool inLowerTierHeavySet(CodeBlock* root)
{
    return s_sampler && s_sampler->inLowerTierHeavySet(root);
}

uint64_t totalSamples(CodeBlock* root)
{
    return s_sampler ? s_sampler->totalSamples(root) : 0;
}

} // namespace TierUpTimeAttribution
} // namespace JSC
