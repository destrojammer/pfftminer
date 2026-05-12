use alloy::primitives::U256;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tiny_keccak::{Hasher, Keccak};

// ─────────────────────────────────────────────────────────────────
// Keccak-256 helpers
// ─────────────────────────────────────────────────────────────────

/// keccak256(challenge[32] ++ nonce_be[32]) — 64-byte fixed input
#[inline(always)]
fn keccak256_64(buf: &[u8; 64]) -> [u8; 32] {
    let mut k = Keccak::v256();
    k.update(buf);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

/// Write `nonce` as big-endian uint64 into bytes [56..64] (upper 24 bytes = 0)
#[inline(always)]
fn set_nonce(buf: &mut [u8; 64], nonce: u64) {
    buf[56..64].copy_from_slice(&nonce.to_be_bytes());
}

/// hash (big-endian 32 bytes) <= target (big-endian 32 bytes)
#[inline(always)]
fn hash_le_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    hash <= target
}

// ─────────────────────────────────────────────────────────────────
// Result returned to caller
// ─────────────────────────────────────────────────────────────────
pub struct PowResult {
    pub nonce:     u64,
    pub hash:      [u8; 32],
    pub elapsed_s: f64,
    pub mhash_s:   f64,
}

// ─────────────────────────────────────────────────────────────────
// CPU Solver — Rayon parallel, work-stealing across all cores
// ─────────────────────────────────────────────────────────────────

/// Find nonce such that keccak256(challenge ++ nonce_be32) ≤ target.
/// `threads` = 0  →  use all logical CPUs.
pub fn solve_cpu(
    challenge: &[u8; 32],
    target:    &[u8; 32],
    threads:   usize,
) -> PowResult {
    let n_threads = if threads == 0 { num_cpus::get() } else { threads };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("rayon pool");

    let found      = Arc::new(AtomicBool::new(false));
    let result_n   = Arc::new(AtomicU64::new(u64::MAX));
    let total_hashes = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let t_ref = Arc::new(start);

    // Each Rayon task covers CHUNK consecutive nonces
    const CHUNK: u64 = 1 << 20; // 1 M per task

    pool.install(|| {
        (0u64..).into_par_iter().find_any(|&chunk_id| {
            if found.load(Ordering::Relaxed) {
                return true;
            }
            let base = chunk_id.wrapping_mul(CHUNK);
            let mut buf = [0u8; 64];
            buf[..32].copy_from_slice(challenge);

            for nonce in base..base.wrapping_add(CHUNK) {
                set_nonce(&mut buf, nonce);
                let hash = keccak256_64(&buf);
                if hash_le_target(&hash, target) {
                    found.store(true, Ordering::Relaxed);
                    result_n.store(nonce, Ordering::Relaxed);
                    return true;
                }
            }
            total_hashes.fetch_add(CHUNK, Ordering::Relaxed);

            // Progress every ~5 M hashes per thread
            let th = total_hashes.load(Ordering::Relaxed);
            if th % (5 * CHUNK * n_threads as u64) < CHUNK {
                let elapsed = t_ref.elapsed().as_secs_f64();
                let mhs     = th as f64 / elapsed / 1_000_000.0;
                eprint!("\r  ⛏️  CPU {mhs:.1} MH/s  [{th:>12} H]  {elapsed:.0}s     ");
            }
            false
        });
    });

    let elapsed_s  = start.elapsed().as_secs_f64();
    let total      = total_hashes.load(Ordering::Relaxed);
    let mhash_s    = total as f64 / elapsed_s / 1_000_000.0;
    let nonce      = result_n.load(Ordering::Relaxed);

    // Recompute hash for the winning nonce
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge);
    set_nonce(&mut buf, nonce);
    let hash = keccak256_64(&buf);

    PowResult { nonce, hash, elapsed_s, mhash_s }
}

// ─────────────────────────────────────────────────────────────────
// GPU Solver — OpenCL (only compiled with --features gpu)
// ─────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::PowResult;
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
        context::Context,
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
        kernel::ExecuteKernel,
        memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
        program::Program,
        types::{cl_ulong, CL_BLOCKING, CL_NON_BLOCKING},
    };
    use std::time::Instant;

    const KECCAK_SRC: &str = r#"
#define KECCAK_ROUNDS 24
__constant ulong RC[24]={
    0x0000000000000001UL,0x0000000000008082UL,0x800000000000808AUL,0x8000000080008000UL,
    0x000000000000808BUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
    0x000000000000008AUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000AUL,
    0x000000008000808BUL,0x800000000000008BUL,0x8000000000008089UL,0x8000000000008003UL,
    0x8000000000008002UL,0x8000000000000080UL,0x000000000000800AUL,0x800000008000000AUL,
    0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL};
__constant int PILN[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__constant int ROTC[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
#define ROL64(x,y)(((x)<<(y))|((x)>>(64-(y))))
void keccakf(ulong st[25]){
    int i,j,r;ulong t,bc[5];
    for(r=0;r<24;r++){
        for(i=0;i<5;i++)bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
        for(i=0;i<5;i++){t=bc[(i+4)%5]^ROL64(bc[(i+1)%5],1);for(j=0;j<25;j+=5)st[j+i]^=t;}
        t=st[1];for(i=0;i<24;i++){j=PILN[i];bc[0]=st[j];st[j]=ROL64(t,ROTC[i]);t=bc[0];}
        for(j=0;j<25;j+=5){for(i=0;i<5;i++)bc[i]=st[j+i];
            for(i=0;i<5;i++)st[j+i]^=(~bc[(i+1)%5])&bc[(i+2)%5];}
        st[0]^=RC[r];}}
void keccak256_64(const uchar*in,uchar*out){
    ulong st[25];for(int i=0;i<25;i++)st[i]=0;
    for(int i=0;i<8;i++){ulong l=0;for(int b=0;b<8;b++)l|=((ulong)in[i*8+b])<<(b*8);st[i]^=l;}
    st[8]^=0x0000000000000001UL;st[16]^=0x8000000000000000UL;
    keccakf(st);
    for(int i=0;i<4;i++){ulong l=st[i];for(int b=0;b<8;b++)out[i*8+b]=(uchar)((l>>(b*8))&0xFF);}}
__kernel void pow_search(
    __global const uchar*challenge,
    ulong nonce_base,
    __global const ulong*target_words,
    __global ulong*result,
    ulong batch_size)
{
    ulong gid=get_global_id(0);
    if(gid>=batch_size||result[0])return;
    ulong nonce=nonce_base+gid;
    uchar inp[64];
    for(int i=0;i<32;i++)inp[i]=challenge[i];
    for(int i=0;i<24;i++)inp[32+i]=0;
    inp[56]=(uchar)((nonce>>56)&0xFF);inp[57]=(uchar)((nonce>>48)&0xFF);
    inp[58]=(uchar)((nonce>>40)&0xFF);inp[59]=(uchar)((nonce>>32)&0xFF);
    inp[60]=(uchar)((nonce>>24)&0xFF);inp[61]=(uchar)((nonce>>16)&0xFF);
    inp[62]=(uchar)((nonce>>8)&0xFF); inp[63]=(uchar)(nonce&0xFF);
    uchar hash[32];keccak256_64(inp,hash);
    ulong hw[4];
    for(int i=0;i<4;i++){hw[i]=0;for(int b=0;b<8;b++)hw[i]=(hw[i]<<8)|hash[i*8+b];}
    bool ok=false;
    for(int i=0;i<4;i++){
        if(hw[i]<target_words[i]){ok=true;break;}
        if(hw[i]>target_words[i])break;
        if(i==3)ok=true;}
    if(ok&&atomic_cmpxchg((volatile __global ulong*)&result[0],0UL,1UL)==0UL)
        result[1]=nonce;
}
"#;

    pub struct GpuSolver {
        context: Context,
        queue:   CommandQueue,
        program: Program,
        name:    String,
    }

    impl GpuSolver {
        pub fn new() -> anyhow::Result<Self> {
            let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
                .map_err(|e| anyhow::anyhow!("OpenCL error: {e}"))?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No GPU found"))?;

            let device  = Device::new(device_id);
            let name    = device.name().unwrap_or_else(|_| "Unknown GPU".into());
            let context = Context::from_device(&device)
                .map_err(|e| anyhow::anyhow!("Context: {e}"))?;
            let queue   = CommandQueue::create_default_with_properties(
                &context, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| anyhow::anyhow!("Queue: {e}"))?;
            let program = Program::create_and_build_from_source(&context, KECCAK_SRC, "")
                .map_err(|e| anyhow::anyhow!("Build: {e}"))?;

            Ok(Self { context, queue, program, name })
        }

        pub fn name(&self) -> &str { &self.name }

        pub fn solve(
            &self,
            challenge:  &[u8; 32],
            target:     &[u8; 32],
            batch_size: usize,
        ) -> anyhow::Result<PowResult> {
            use opencl3::kernel::Kernel;

            let kernel = Kernel::create(&self.program, "pow_search")
                .map_err(|e| anyhow::anyhow!("Kernel: {e}"))?;

            // Convert target to 4 × u64 big-endian words
            let mut tw = [0u64; 4];
            for i in 0..4 {
                tw[i] = u64::from_be_bytes(target[i*8..i*8+8].try_into().unwrap());
            }

            let ch_buf: Buffer<u8> = Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                32,
                challenge.as_ptr() as *mut _)
                .map_err(|e| anyhow::anyhow!("Buffer ch: {e}"))?;

            let tgt_buf: Buffer<u64> = Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                4,
                tw.as_ptr() as *mut _)
                .map_err(|e| anyhow::anyhow!("Buffer tgt: {e}"))?;

            let mut result = [0u64; 2];
            let res_buf: Buffer<u64> = Buffer::create(
                &self.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                2,
                result.as_mut_ptr() as *mut _)
                .map_err(|e| anyhow::anyhow!("Buffer res: {e}"))?;

            let start = Instant::now();
            let mut nonce_base = 0u64;
            let mut total: u64 = 0;
            let mut last_report = start;

            loop {
                // Reset result buffer
                result = [0u64; 2];
                self.queue
                    .enqueue_write_buffer(&res_buf, CL_BLOCKING, 0, &result, &[])
                    .map_err(|e| anyhow::anyhow!("Write buf: {e}"))?;

                ExecuteKernel::new(&kernel)
                    .set_arg(&ch_buf)
                    .set_arg(&(nonce_base as cl_ulong))
                    .set_arg(&tgt_buf)
                    .set_arg(&res_buf)
                    .set_arg(&(batch_size as cl_ulong))
                    .set_global_work_size(batch_size)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|e| anyhow::anyhow!("Enqueue: {e}"))?;
                self.queue.finish().map_err(|e| anyhow::anyhow!("Finish: {e}"))?;

                self.queue
                    .enqueue_read_buffer(&res_buf, CL_BLOCKING, 0, &mut result, &[])
                    .map_err(|e| anyhow::anyhow!("Read buf: {e}"))?;

                total += batch_size as u64;

                if result[0] == 1 {
                    let nonce     = result[1];
                    let elapsed_s = start.elapsed().as_secs_f64();
                    let mhash_s   = total as f64 / elapsed_s / 1_000_000.0;

                    // Verify locally
                    let mut buf = [0u8; 64];
                    buf[..32].copy_from_slice(challenge);
                    buf[56..64].copy_from_slice(&nonce.to_be_bytes());
                    use tiny_keccak::{Hasher, Keccak};
                    let mut k = Keccak::v256();
                    k.update(&buf);
                    let mut hash = [0u8; 32];
                    k.finalize(&mut hash);

                    return Ok(PowResult { nonce, hash, elapsed_s, mhash_s });
                }

                let now = Instant::now();
                if now.duration_since(last_report).as_secs_f64() >= 3.0 {
                    let elapsed = now.duration_since(start).as_secs_f64();
                    let mhs     = total as f64 / elapsed / 1_000_000.0;
                    eprint!("\r  ⛏️  GPU {mhs:.1} MH/s  [{total:>12} H]  {elapsed:.0}s     ");
                    last_report = now;
                }

                nonce_base = nonce_base.wrapping_add(batch_size as u64);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Benchmark (2-second hash-rate test, CPU only)
// ─────────────────────────────────────────────────────────────────
pub fn benchmark_cpu() -> f64 {
    let challenge = [0u8; 32];
    let start     = Instant::now();
    let deadline  = std::time::Duration::from_secs(2);
    let count     = Arc::new(AtomicU64::new(0));
    let count2    = count.clone();
    let n         = num_cpus::get();

    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
        .install(|| {
            (0u64..).into_par_iter().find_any(|&chunk| {
                if start.elapsed() >= deadline { return true; }
                let base = chunk.wrapping_mul(1 << 16);
                let mut buf = [0u8; 64];
                buf[..32].copy_from_slice(&challenge);
                for nonce in base..base.wrapping_add(1 << 16) {
                    set_nonce(&mut buf, nonce);
                    let _ = keccak256_64(&buf);
                }
                count2.fetch_add(1 << 16, Ordering::Relaxed);
                false
            });
        });

    let elapsed = start.elapsed().as_secs_f64();
    count.load(Ordering::Relaxed) as f64 / elapsed / 1_000_000.0
}
