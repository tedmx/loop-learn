use anyhow::{Context, Result};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::token::LlamaToken;
use crate::loader::ModelFiles;
use crate::storage::ChatSession;
use crate::presets::PresetInfo; 

pub struct InferenceEngine {
    model: LlamaModel,
    backend: LlamaBackend,
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

        Ok(Self {
            model,
            backend,
        })
    }

    pub fn generate<'a, F>(
        &self,
        ctx: &mut llama_cpp_2::context::LlamaContext<'a>,
        prompt: &str,
        session: &mut ChatSession,
        preset: &crate::presets::PresetInfo,
        mut on_token: F,
    ) -> Result<()>where
        F: FnMut(&str) -> Result<()>,
    {

        let tokens_list = self.model.str_to_token(prompt, llama_cpp_2::model::AddBos::Always)?;
        let n_tokens_all = tokens_list.len();

        ctx.clear_kv_cache();
        
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(2048, 1);
        let mut current_pos = 0i32;

        let (prompt_init, prompt_tail) = tokens_list.split_at(n_tokens_all - 1);
        let last_prompt_token = prompt_tail[0];

        let chunk_size = 512;
        for chunk in prompt_init.chunks(chunk_size) {
            batch.clear();
            for token in chunk {
                batch.add(*token, current_pos, &[0], false)?;
                current_pos += 1;
            }
            ctx.decode(&mut batch).context("Failed to decode prompt execution chunk")?;
        }

        batch.clear();
        batch.add(last_prompt_token, current_pos, &[0], true)?;
        ctx.decode(&mut batch).context("Failed to decode prompt trailing token")?;
        current_pos += 1;

        let mut prev_text = String::new();

        // Extract the first token logits from the prompt residue before altering the batch
        let first_logit_idx = if batch.n_tokens() > 0 {
            batch.n_tokens() - 1
        } else {
            0
        };

        let first_candidates = ctx.candidates_ith(first_logit_idx);
        let mut next_token_id = first_candidates
            .max_by(|a, b| a.logit().partial_cmp(&b.logit()).unwrap())
            .map(|c| c.id().0)
            .unwrap_or(0);

        let mut token_count = 0;
        // Single linear autoregressive generation loop
        loop {
            if next_token_id == preset.eos_token as i32 {
                break;
            }
            if token_count >= preset.max_tokens {
                break;
            }

            let next_token = LlamaToken(next_token_id);

            let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
            
            if let Ok(token_text) = self.model.token_to_piece(
                next_token, 
                &mut decoder, 
                true,
                None 
            ) {
                if token_text == "<|im_end|>" || token_text == "<|end|>" || token_text == "<|endoftext|>" {
                    break;
                }
                
                on_token(&token_text)?;
                prev_text.push_str(&token_text);
            }

            // Wipe prompt leftovers and stage exactly one token at index 0
            batch.clear();
            batch.add(next_token, current_pos, &[0], true)?;
            
            ctx.decode(&mut batch).context(format!("Decode failed at pos {}", current_pos))?;

            current_pos += 1;

            next_token_id = ctx.candidates_ith(0)
                .max_by(|a, b| a.logit().partial_cmp(&b.logit()).unwrap())
                .map(|c| c.id().0)
                .unwrap_or(0);
            
            token_count += 1;
        }

        let clean_final_text = prev_text.replace("<br>", "\n");
        session.add_message("assistant", clean_final_text.trim());

        Ok(())
    }

    // Allocate persistent execution context space inside GPU VRAM
    pub fn new_context(&self, preset: &PresetInfo) -> Result<llama_cpp_2::context::LlamaContext<'_>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_batch(512)
            .with_n_ctx(std::num::NonZeroU32::new(preset.n_ctx));

        let ctx = self.model.new_context(&self.backend, ctx_params)
            .context("Failed to instantiate isolated execution context space")?;
        Ok(ctx)
    }
}