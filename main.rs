mod contract;
mod pow;
mod wallet;

use anyhow::{Context, Result};
use clap::Parser;
use std::{path::PathBuf, sync::atomic::{AtomicBool, Ordering}, sync::Arc, time::Instant};

// ─────────────────────────────────────────────────────────────────
// CLI Arguments
// ─────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(
    name    = "pfft-miner",
    version = "0.2.0",
    about   = "⛏️  PFFT GPU/CPU Miner — Rust Edition",
    long_about = None
)]
struct Args {
    /// Ethereum RPC endpoint
    #[arg(long, env = "ETH_RPC",
          default_value = "https://ethereum-rpc.publicnode.com")]
    rpc: String,

    /// Path to wallet JSON file
    #[arg(long, env = "PFFT_WALLET", default_value = "wallet.json")]
    wallet: PathBuf,

    /// CPU thread count (0 = all cores)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// GPU batch size in hashes per dispatch (GPU mode only)
    #[arg(long, default_value = "4194304")]
    gpu_batch: usize,

    /// Force CPU mode even if GPU is available
    #[arg(long, default_value = "false")]
    cpu_only: bool,

    /// Run a quick benchmark and exit
    #[arg(long, default_value = "false")]
    benchmark: bool,

    /// Seconds to pause between mint rounds
    #[arg(long, default_value = "5")]
    pause: u64,
}

// ─────────────────────────────────────────────────────────────────
// Pretty display helpers
// ─────────────────────────────────────────────────────────────────
fn fmt_pfft(wei: &alloy::primitives::U256) -> f64 {
    let s: u128 = (*wei / alloy::primitives::U256::from(10u128.pow(18))).to();
    s as f64
}

fn fmt_pfft_f(wei: &alloy::primitives::U256) -> f64 {
    // with 2-decimal precision
    let n: u128 = wei.to();
    n as f64 / 1e18
}

fn sep(c: char, n: usize) { println!("{}", c.to_string().repeat(n)); }

// ─────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<()> {
    // Load .env if present
    let _ = dotenvy::dotenv();

    let args = Args::parse();

    sep('=', 62);
    println!("  ⛏️  PFFT Miner — Rust Edition");
    println!("  Contract : {}", contract::CONTRACT_ADDR);
    println!("  RPC      : {}", args.rpc);
    sep('=', 62);

    // ── Benchmark mode ───────────────────────────────────────────
    if args.benchmark {
        println!("\n🔬 Benchmark CPU ({} cores)...", num_cpus::get());
        let mhs = pow::benchmark_cpu();
        println!("  📊 Hasil: {mhs:.2} MH/s");
        return Ok(());
    }

    // ── GPU detection ────────────────────────────────────────────
    #[cfg(feature = "gpu")]
    let gpu_solver: Option<pow::gpu::GpuSolver> = if args.cpu_only {
        println!("\n🔧 Mode: CPU (--cpu-only)");
        None
    } else {
        match pow::gpu::GpuSolver::new() {
            Ok(g) => {
                println!("\n🎮 GPU ditemukan : {}", g.name());
                println!("   Batch size    : {} hashes/dispatch", args.gpu_batch);
                Some(g)
            }
            Err(e) => {
                println!("\n⚠️  GPU tidak tersedia ({e}) → fallback ke CPU");
                None
            }
        }
    };

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n🔧 Mode: CPU ({} cores, GPU feature tidak dikompilasi)",
                 num_cpus::get());
        println!("   Untuk GPU: cargo build --release --features gpu");
    }

    #[cfg(feature = "gpu")]
    let use_gpu = gpu_solver.is_some();
    #[cfg(not(feature = "gpu"))]
    let use_gpu = false;

    if !use_gpu {
        let n = if args.threads == 0 { num_cpus::get() } else { args.threads };
        println!("🔧 CPU threads   : {n}");
        let mhs = pow::benchmark_cpu();
        println!("📊 CPU benchmark : {mhs:.1} MH/s");
    }

    // ── Ctrl-C handler ───────────────────────────────────────────
    let running = Arc::new(AtomicBool::new(true));
    let r2      = running.clone();
    ctrlc::set_handler(move || {
        println!("\n\n  ⚠️  Ctrl-C — menghentikan miner...");
        r2.store(false, Ordering::SeqCst);
    }).context("ctrlc handler")?;

    // ── Wallet ───────────────────────────────────────────────────
    println!();
    let pk_hex = if args.wallet.exists() {
        let pk = wallet::load_wallet(&args.wallet)?;
        println!("✅ Wallet dimuat  : {}", args.wallet.display());
        pk
    } else {
        println!("🆕 Wallet tidak ada — membuat baru...");
        wallet::create_wallet(&args.wallet)?
    };

    // ── Provider + Contract ──────────────────────────────────────
    let (provider, wallet_addr, contract) =
        contract::make_provider(&args.rpc, &pk_hex)?;

    // Check connection
    let block = provider.get_block_number().await
        .context("Cannot connect to RPC")?;
    println!("✅ Terhubung      : Block #{block}");
    println!("✅ Wallet address : {wallet_addr:?}");

    // ETH balance
    let eth_bal = provider.get_balance(wallet_addr).await?;
    let eth_f   = eth_bal.to::<u128>() as f64 / 1e18;
    println!("💰 ETH balance   : {eth_f:.6}");
    if eth_f < 0.00005 {
        println!("⚠️  ETH rendah! Butuh minimal ~0.00005 ETH untuk gas.");
    }

    // ── Initial contract status ───────────────────────────────────
    let s = contract::get_status(&contract, wallet_addr).await?;
    println!("\n📊 Status Kontrak:");
    println!("   Minted      : {:.0} / {:.0} PFFT ({:.1}%)",
             fmt_pfft_f(&s.total_minted),
             fmt_pfft_f(&s.max_supply),
             s.progress_pct);
    println!("   Next mint   : ~{:.2} PFFT", fmt_pfft_f(&s.next_mint));
    println!("   Difficulty  : {} hex zeros ({}-bit)", s.hex_zeros, s.difficulty_bits);
    println!("   Wallet mint : {:.2} / 10,000 PFFT", fmt_pfft_f(&s.wallet_minted));
    println!("   Wallet bal  : {:.2} PFFT", fmt_pfft_f(&s.wallet_bal));

    // ── Mining loop ──────────────────────────────────────────────
    let mut round_num    = 0u64;
    let mut total_mints  = 0u64;
    let mut total_pfft   = 0.0f64;
    let session_start    = Instant::now();

    while running.load(Ordering::SeqCst) {
        round_num += 1;
        sep('─', 62);
        println!("  Round #{round_num}");
        sep('─', 62);

        // Refresh status
        let s = match contract::get_status(&contract, wallet_addr).await {
            Ok(s) => s,
            Err(e) => {
                println!("  ⚠️  Status error: {e} — retry 15s...");
                tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
                continue;
            }
        };

        println!("  Supply    : {:.0} ({:.1}%) | Next ~{:.2} PFFT | Diff {}-bit",
                 fmt_pfft_f(&s.total_minted),
                 s.progress_pct,
                 fmt_pfft_f(&s.next_mint),
                 s.difficulty_bits);

        if s.total_minted >= s.max_supply {
            println!("  🏁 Max supply tercapai — selesai!");
            break;
        }
        if s.wallet_minted >= alloy::primitives::U256::from(10_000u128) *
           alloy::primitives::U256::from(10u128.pow(18)) {
            println!("  🏁 Wallet cap 10,000 PFFT tercapai — selesai!");
            break;
        }

        // Get challenge
        let challenge = match contract::get_challenge(&contract, wallet_addr).await {
            Ok(c) => c,
            Err(e) => {
                println!("  ⚠️  Challenge error: {e} — retry...");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        // Convert U256 target to [u8; 32]
        let mut target_bytes = [0u8; 32];
        s.target.to_big_endian(&mut target_bytes);

        // ── Solve PoW ──────────────────────────────────────────
        let mode_str = if use_gpu { "GPU" } else { "CPU" };
        println!("  ⛏️  Mining ({mode_str}, {}-bit)...", s.difficulty_bits);

        let result = {
            #[cfg(feature = "gpu")]
            if let Some(ref gpu) = gpu_solver {
                // GPU solve (blocking, runs on rayon-internal thread to avoid blocking tokio)
                let ch  = challenge;
                let tgt = target_bytes;
                let bat = args.gpu_batch;
                tokio::task::spawn_blocking(move || {
                    gpu.solve(&ch, &tgt, bat)
                }).await
                 .context("GPU task")??
            } else {
                let ch  = challenge;
                let tgt = target_bytes;
                let thr = args.threads;
                tokio::task::spawn_blocking(move || {
                    pow::solve_cpu(&ch, &tgt, thr)
                }).await
                 .context("CPU task")?
            }

            #[cfg(not(feature = "gpu"))]
            {
                let ch  = challenge;
                let tgt = target_bytes;
                let thr = args.threads;
                tokio::task::spawn_blocking(move || {
                    pow::solve_cpu(&ch, &tgt, thr)
                }).await
                 .context("CPU task")?
            }
        };

        if !running.load(Ordering::SeqCst) { break; }

        println!("\n  ✅ Nonce ditemukan : {}", result.nonce);
        println!("  ⚡ Waktu mining    : {:.1}s  |  {:.2} MH/s",
                 result.elapsed_s, result.mhash_s);
        println!("  🔑 Hash            : 0x{}", hex::encode(result.hash));

        // On-chain verify before submit
        match contract::verify_pow(&contract, wallet_addr, result.nonce).await {
            Ok(true)  => {}
            Ok(false) => {
                println!("  ⚠️  Nonce invalid on-chain (supply berubah?), re-mine...");
                continue;
            }
            Err(e) => {
                println!("  ⚠️  Verify error: {e}, submit saja...");
            }
        }

        // Submit transaction
        println!("  📤 Submit freeMint({})...", result.nonce);
        match contract::submit_mint(&contract, result.nonce).await {
            Ok(tx_hash) => {
                println!("  ✅ MINT OK  tx: https://etherscan.io/tx/{tx_hash}");
                total_mints += 1;
                let earned   = fmt_pfft_f(&s.next_mint);
                total_pfft  += earned;
                println!("  💰 +{earned:.2} PFFT  |  Total: {total_pfft:.2} PFFT dari {total_mints} mint");

                // Refresh balance
                if let Ok(fresh) = contract::get_status(&contract, wallet_addr).await {
                    println!("  💰 Balance: {:.2} PFFT", fmt_pfft_f(&fresh.wallet_bal));
                }
            }
            Err(e) => {
                println!("  ❌ TX error: {e}");
            }
        }

        // Session summary
        let elapsed = session_start.elapsed().as_secs_f64();
        println!("\n  📈 Sesi: {total_mints} mint | {total_pfft:.2} PFFT | {:.1} menit",
                 elapsed / 60.0);

        // Cooldown
        if running.load(Ordering::SeqCst) && args.pause > 0 {
            println!("  ⏳ Cooldown {}s...", args.pause);
            tokio::time::sleep(tokio::time::Duration::from_secs(args.pause)).await;
        }
    }

    // ── Final summary ─────────────────────────────────────────────
    sep('=', 62);
    println!("  Ringkasan Sesi");
    println!("  Mint       : {total_mints}");
    println!("  PFFT diraih: {total_pfft:.2}");
    println!("  Runtime    : {:.1} menit",
             session_start.elapsed().as_secs_f64() / 60.0);
    sep('=', 62);

    Ok(())
}
