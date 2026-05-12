# PFFT Miner — Rust Edition ⛏️

Full Rust binary miner untuk **Pow Free Fair Token (PFFT)** di Ethereum Mainnet.
Tidak ada Python, tidak ada dependency runtime — satu binary standalone.

## Struktur Project

```
pfft-miner/
├── Cargo.toml
└── src/
    ├── main.rs      ← Entry point + CLI + mining loop
    ├── contract.rs  ← Ethereum ABI, RPC calls, tx submit
    ├── pow.rs       ← PoW solver (Rayon CPU + OpenCL GPU)
    └── wallet.rs    ← Wallet create / load / save
```

## Build

```bash
# Install Rust (jika belum)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# CPU only (semua platform)
cargo build --release

# CPU + OpenCL GPU (NVIDIA / AMD / Intel)
cargo build --release --features gpu

# Binary ada di:
./target/release/pfft-miner
```

## Cara Pakai

```bash
# Jalankan (auto create wallet jika belum ada)
./pfft-miner

# Custom RPC
./pfft-miner --rpc https://mainnet.infura.io/v3/YOUR_KEY

# Custom wallet path
./pfft-miner --wallet /path/to/wallet.json

# Set jumlah CPU thread (default: semua core)
./pfft-miner --threads 8

# GPU batch size (default: 4194304 = 4M hashes)
./pfft-miner --gpu-batch 8388608

# Force CPU meski ada GPU
./pfft-miner --cpu-only

# Benchmark hashrate CPU
./pfft-miner --benchmark

# Pause antar round (detik)
./pfft-miner --pause 10
```

### Environment Variables

Bisa juga pakai `.env` file:

```env
ETH_RPC=https://ethereum-rpc.publicnode.com
PFFT_WALLET=./wallet.json
```

## Engine Priority

```
GPU (OpenCL, --features gpu)  →  Rust Rayon CPU  →  CPU (--cpu-only)
```

## Performa Estimasi

| Hardware          | Mode          | Hashrate         |
|-------------------|---------------|------------------|
| Ryzen 5 5600X     | Rust CPU      | ~30–50 MH/s      |
| i9-13900K         | Rust CPU      | ~80–120 MH/s     |
| RTX 3060          | Rust GPU OCL  | ~200–350 MH/s    |
| RTX 4070          | Rust GPU OCL  | ~400–600 MH/s    |
| RX 6700 XT        | Rust GPU OCL  | ~250–400 MH/s    |

## Keamanan

- `wallet.json` berisi private key — **JANGAN di-commit ke git**
- File otomatis di-`chmod 600` saat dibuat
- Jangan share private key ke siapapun

## Troubleshooting

**OpenCL tidak ditemukan:**
```bash
# NVIDIA — install CUDA atau OpenCL ICD
sudo apt install ocl-icd-opencl-dev nvidia-opencl-dev

# AMD — install ROCm
sudo apt install rocm-opencl-dev

# Cek device:
clinfo
```

**RPC error / timeout:**
Gunakan RPC premium seperti Infura, Alchemy, atau QuickNode untuk stabilitas lebih baik.
