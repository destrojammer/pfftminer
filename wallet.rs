use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fs, os::unix::fs::PermissionsExt, path::Path};

#[derive(Debug, Serialize, Deserialize)]
pub struct WalletFile {
    pub address:         String,
    pub private_key_hex: String,
    pub created:         String,
    pub note:            String,
}

/// Load wallet from JSON file, return hex private key (no 0x prefix).
pub fn load_wallet(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("Cannot read wallet file: {}", path.display()))?;
    let wf: WalletFile = serde_json::from_str(&raw)
        .context("Wallet JSON parse error")?;
    let pk = wf.private_key_hex.trim_start_matches("0x").to_string();
    Ok(pk)
}

/// Create a new random wallet, persist it, chmod 600. Returns hex private key.
pub fn create_wallet(path: &Path) -> Result<String> {
    use alloy::signers::local::PrivateKeySigner;
    use rand::rngs::OsRng;

    let signer = PrivateKeySigner::random_with(&mut OsRng);
    let address = format!("{:?}", signer.address());
    let pk_hex  = hex::encode(signer.credential().to_bytes());

    let wf = WalletFile {
        address:         address.clone(),
        private_key_hex: pk_hex.clone(),
        created:         chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        note:            "PFFT miner wallet — JAGA KERAHASIAANNYA!".into(),
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(&wf)?)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;

    println!("  ✅ Wallet baru  : {address}");
    println!("  💾 Disimpan di  : {}", path.display());

    Ok(pk_hex)
}
