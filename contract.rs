use alloy::{
    contract::ContractInstance,
    dyn_abi::DynSolValue,
    network::{EthereumWallet, TransactionBuilder},
    primitives::{Address, U256},
    providers::{Provider, ProviderBuilder},
    rpc::types::TransactionRequest,
    signers::local::PrivateKeySigner,
    sol,
    transports::http::reqwest::Url,
};
use anyhow::{Context, Result};
use std::str::FromStr;

pub const CONTRACT_ADDR: &str = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB";

// ── Solidity ABI via alloy sol! macro ────────────────────────────
sol! {
    #[sol(rpc)]
    interface IPfft {
        function currentPowHexZeros() external view returns (uint256);
        function totalMinted()        external view returns (uint256);
        function MAX_SUPPLY()         external view returns (uint256);
        function calculateActualMint(uint256 requested) external view returns (uint256);
        function currentPowChallenge(address user)      external view returns (bytes32);
        function isValidPow(address user, uint256 powNonce) external view returns (bool);
        function freeMint(uint256 powNonce) external;
        function mintedByAddress(address user) external view returns (uint256);
        function balanceOf(address account)    external view returns (uint256);
    }
}

// ── Shared types ─────────────────────────────────────────────────
pub struct ContractStatus {
    pub hex_zeros:      u64,
    pub difficulty_bits: u64,
    pub total_minted:   U256,
    pub max_supply:     U256,
    pub next_mint:      U256,
    pub wallet_minted:  U256,
    pub wallet_bal:     U256,
    pub target:         U256,
    pub progress_pct:   f64,
}

// ── Provider / Contract helpers ───────────────────────────────────

pub fn make_provider(
    rpc_url: &str,
    private_key_hex: &str,
) -> Result<(
    impl Provider + Clone,
    Address,
    IPfft::IPfftInstance<_, _>,
)> {
    let url    = Url::parse(rpc_url).context("Invalid RPC URL")?;
    let signer: PrivateKeySigner = private_key_hex.parse()
        .context("Invalid private key")?;
    let address = signer.address();
    let wallet  = EthereumWallet::from(signer);

    let provider = ProviderBuilder::new()
        .with_recommended_fillers()
        .wallet(wallet)
        .on_http(url);

    let contract = IPfft::new(
        Address::from_str(CONTRACT_ADDR).unwrap(),
        provider.clone(),
    );

    Ok((provider, address, contract))
}

/// Fetch all relevant contract + wallet state in parallel (multiple calls).
pub async fn get_status(
    contract: &IPfft::IPfftInstance<impl Provider + Clone, impl alloy::transports::Transport + Clone>,
    wallet_addr: Address,
) -> Result<ContractStatus> {
    // Fire all calls concurrently
    let (hz, tot, mx, wm, wb) = tokio::try_join!(
        async { contract.currentPowHexZeros().call().await.map(|r| r._0) },
        async { contract.totalMinted().call().await.map(|r| r._0) },
        async { contract.MAX_SUPPLY().call().await.map(|r| r._0) },
        async { contract.mintedByAddress(wallet_addr).call().await.map(|r| r._0) },
        async { contract.balanceOf(wallet_addr).call().await.map(|r| r._0) },
    )?;

    let requested = U256::from(1_000u64) * U256::from(10u64).pow(U256::from(18u64)); // 1000 ether
    let nxt = contract.calculateActualMint(requested).call().await?._0;

    let hex_zeros       = hz.to::<u64>();
    let difficulty_bits = hex_zeros * 4;

    // target = (2^256 - 1) >> (hex_zeros * 4)
    let target = if difficulty_bits >= 256 {
        U256::ZERO
    } else {
        U256::MAX >> difficulty_bits
    };

    let progress_pct = if mx.is_zero() {
        0.0
    } else {
        let t: u128 = tot.to();
        let m: u128 = mx.to();
        t as f64 * 100.0 / m as f64
    };

    Ok(ContractStatus {
        hex_zeros,
        difficulty_bits,
        total_minted: tot,
        max_supply:   mx,
        next_mint:    nxt,
        wallet_minted: wm,
        wallet_bal:   wb,
        target,
        progress_pct,
    })
}

/// Get the current PoW challenge bytes for a wallet address.
pub async fn get_challenge(
    contract: &IPfft::IPfftInstance<impl Provider + Clone, impl alloy::transports::Transport + Clone>,
    wallet_addr: Address,
) -> Result<[u8; 32]> {
    let b32 = contract.currentPowChallenge(wallet_addr).call().await?.challenge;
    Ok(b32.0)
}

/// On-chain verify that the found nonce is still valid before submitting.
pub async fn verify_pow(
    contract: &IPfft::IPfftInstance<impl Provider + Clone, impl alloy::transports::Transport + Clone>,
    wallet_addr: Address,
    nonce: u64,
) -> Result<bool> {
    let valid = contract
        .isValidPow(wallet_addr, U256::from(nonce))
        .call()
        .await?
        ._0;
    Ok(valid)
}

/// Submit freeMint(nonce) transaction. Returns tx hash string on success.
pub async fn submit_mint(
    contract: &IPfft::IPfftInstance<impl Provider + Clone, impl alloy::transports::Transport + Clone>,
    nonce: u64,
) -> Result<String> {
    let tx_hash = contract
        .freeMint(U256::from(nonce))
        .gas(200_000u64)
        .send()
        .await
        .context("send freeMint tx")?
        .watch()
        .await
        .context("wait for receipt")?;

    Ok(format!("{tx_hash:?}"))
}
