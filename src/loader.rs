use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use std::io::Write;
use crate::presets::PresetInfo;

pub struct ModelFiles {
    pub model_path: PathBuf,
}

impl ModelFiles {
    pub fn download_model_files(preset: &PresetInfo) -> Result<Self> {
        println!("Checking local cache for GGUF model: {}/{}", preset.repo_id, preset.filename);
        std::io::stdout().flush()?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .context("Failed to initialize Hugging Face API Client")?;

        let repo = api.model(preset.repo_id.to_string());
        let model_path = repo.get(preset.filename)
            .context("Failed to download or locate GGUF model weight file")?;

        println!("GGUF file verified and ready to use.");

        Ok(ModelFiles { model_path })
    }
}