use anyhow::{Context, Result};
use std::io::Write;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::token::LlamaToken;
use crate::loader::ModelFiles;
use crate::storage::ChatSession;

pub struct InferenceEngine {
    model: LlamaModel,
    backend: LlamaBackend,
    eos_token: u32,
}

impl InferenceEngine {
    pub fn new(files: &ModelFiles, _cuda_index: i32) -> Result<Self> {
        println!("Initializing unified backend infrastructure...");

        let mut backend = LlamaBackend::init()?;
        backend.void_logs();

        let model_params = LlamaModelParams::default();
        let model_path = &files.model_path;

        println!("Loading target GGUF model into hardware context...");
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("Failed to build LlamaModel structure from storage binary")?;

        let eos_token = model.token_eos().0 as u32;

        Ok(Self {
            model,
            backend,
            eos_token,
        })
    }

    pub fn generate(&self, prompt: &str, session: &mut ChatSession) -> Result<()> {
        let ctx_params = LlamaContextParams::default()
            .with_n_batch(2048)
            .with_n_ctx(std::num::NonZeroU32::new(4096));

        let n_batch = ctx_params.n_batch();
        let n_ctx_configured = ctx_params.n_ctx();
        let n_batch_configured = ctx_params.n_batch();

        let mut ctx = self.model.new_context(&self.backend, ctx_params)
            .context("Failed to instantiate isolated execution context space")?;

        let tokens_list = self.model.str_to_token(prompt, llama_cpp_2::model::AddBos::Always)?;

        // --- DEBUG ROOT CAUSE BLOCK START ---
        let n_tokens_all = tokens_list.len();
        println!("[DEBUG] Llama core decoding execution constraints check:");
        println!("[DEBUG] Total incoming tokens count (n_tokens_all): {}", n_tokens_all);
        println!("[DEBUG] Configured context batch threshold (n_batch): {}", n_batch);
    
        let model_ctx_train: u32 = self.model.n_ctx_train();
        println!("[DEBUG] Model trained max context (n_ctx_train): {}", model_ctx_train);
        println!("[DEBUG] Runtime configured context size (n_ctx): {:?}", n_ctx_configured);
        println!("[DEBUG] Runtime configured batch size (n_batch): {}", n_batch_configured);
        // --- DEBUG ROOT CAUSE BLOCK END ---
        
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(2048, 1);
        let mut current_pos = 0i32;

        let chunk_size = 512;
        for chunk in tokens_list.chunks(chunk_size) {
            batch.clear();
            
            for (i, token) in chunk.iter().enumerate() {
                let is_last = i == chunk.len() - 1;
                batch.add(*token, current_pos, &[0], is_last)?;
                current_pos += 1;
            }
            
            ctx.decode(&mut batch).context("Failed to decode prompt execution chunk")?;
        }

        let mut prev_text = String::new();

        loop {
            let logits = ctx.candidates_ith(batch.n_tokens() - 1);
            let logits_vec = logits.map(|c| c.logit()).collect::<Vec<f32>>();

            let temperature = 0.7f32;
            let max_logit = logits_vec.iter().copied().fold(f32::NAN, f32::max);
            
            let exps: Vec<f32> = logits_vec
                .iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .collect();
            
            let sum_exps: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

            let mut rng_seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let random_val = (rng_seed >> 32) as f32 / (u32::MAX as f32);

            let mut cumulative_prob = 0.0f32;
            let mut next_token_id = 0;
            
            for (idx, &prob) in probs.iter().enumerate() {
                cumulative_prob += prob;
                if random_val <= cumulative_prob {
                    next_token_id = idx as i32;
                    break;
                }
                if idx == probs.len() - 1 {
                    next_token_id = idx as i32;
                }
            }

            let next_token = LlamaToken(next_token_id);
            
            if let Ok(token_text) = self.model.token_to_str(next_token, llama_cpp_2::model::Special::Tokenize) {
                if token_text == "<|im_end|>" || token_text == "<|end|>" {
                    break;
                }
                print!("{}", token_text);
                std::io::stdout().flush()?;
                prev_text.push_str(&token_text);
            }

            batch.clear();
            batch.add(next_token, current_pos, &[0], true)?;
            
            ctx.decode(&mut batch).context("Failed to decode subsequent token step")?;
            current_pos += 1;
        }

        println!();

        let clean_final_text = prev_text.replace("<br>", "\n");
        session.add_message("assistant", clean_final_text.trim());
        session.pos_offset = current_pos as usize;

        Ok(())
    }
}