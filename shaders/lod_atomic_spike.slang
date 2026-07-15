// LOD int64 BDA-atomics feasibility spike (Task 1, GATING). Proves that a 64-bit integer
// atomic-add through a buffer-device-address (PhysicalStorageBuffer) pointer compiles and
// produces a correct, deterministic result on this Slang/driver/GPU combination. Env-gated
// (MIMIR_LOD_ATOMIC_SPIKE) and NOT part of any render path -- it exists only to de-risk the
// LOD-centroid position-sum accumulator. The validated atomic form below is reused verbatim
// by the LOD-centroid scatter pass to accumulate the per-cell int64 position sums.
struct PushConstants
{
    uint64_t* sink; // BDA pointer to a single uint64 accumulator (offset 0). 8-byte aligned.
    uint addend;    // value each thread atomically adds
    uint count;     // number of dispatched threads that contribute
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void spikeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= pc.count) { return; }

    // 64-bit OpAtomicIAdd through a PhysicalStorageBuffer pointer. InterlockedAdd on the
    // dereferenced uint64_t* lowers to `OpAtomicIAdd %ulong <ptr> Device None` (Int64Atomics
    // capability) -- verified in the emitted SPIR-V. This is the pattern Task 2 reuses; when the
    // accumulator has many elements, take the address explicitly in 64-bit integer arithmetic
    // (uint64_t(base) + idx*8) and reinterpret to uint64_t*, NEVER pointer-index (NVIDIA truncates
    // OpPtrAccessChain offsets past 4 GiB; see pathtrace_aabbs.slang). Single element here => offset 0.
    uint64_t* p = pc.sink;
    uint64_t old;
    InterlockedAdd(*p, uint64_t(pc.addend), old);
}
