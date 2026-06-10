use anyhow::{Context, Result};
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

fn sample_greedy(logits_vec: &[f32], _preset: &crate::presets::PresetInfo, _rng_seed: &mut u64) -> i32 {
    let mut next_token_id = 0i32;
    let mut max_logit_val = -f32::INFINITY;

    for (idx, &logit) in logits_vec.iter().enumerate() {
        if logit > max_logit_val {
            max_logit_val = logit;
            next_token_id = idx as i32;
        }
    }

    next_token_id
}

fn sample_top_p(logits_vec: &[f32], preset: &crate::presets::PresetInfo, rng_seed: &mut u64) -> i32 {
    let temperature = preset.temperature as f32;
    let max_logit = logits_vec.iter().copied().filter(|&l| l != -f32::INFINITY).fold(f32::NAN, f32::max);
    
    let max_logit = if max_logit.is_nan() { 0.0f32 } else { max_logit };
    
    let exps: Vec<f32> = logits_vec
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .collect();
    
    let sum_exps: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();
    
    let mut sorted_probs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_p = preset.top_p as f32;
    let mut cumulative_prob = 0.0f32;
    let mut cutoff_index = sorted_probs.len();

    for (i, &(_, prob)) in sorted_probs.iter().enumerate() {
        cumulative_prob += prob;
        if cumulative_prob >= top_p {
            cutoff_index = std::cmp::min(i + 1, sorted_probs.len());
            break;
        }
    }
    sorted_probs.truncate(cutoff_index);

    let truncated_sum: f32 = sorted_probs.iter().map(|&(_, p)| p).sum();
    for (_, p) in sorted_probs.iter_mut() {
        *p /= truncated_sum;
    }

    *rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let random_val = (*rng_seed >> 32) as f32 / (u32::MAX as f32);

    let mut sample_cum_prob = 0.0f32;
    let mut next_token_id = sorted_probs.last().unwrap().0 as i32;
    
    for &(idx, prob) in &sorted_probs {
        sample_cum_prob += prob;
        if random_val <= sample_cum_prob {
            next_token_id = idx as i32;
            break;
        }
    }

    next_token_id
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

    pub fn generate<F>(
        &self,
        ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
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
        let mut generated_tokens = tokens_list.clone();

        let mut rng_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Extract the first token logits from the prompt residue before altering the batch
        let first_logit_idx = if batch.n_tokens() > 0 {
            batch.n_tokens() - 1
        } else {
            0
        };

        let first_candidates = ctx.candidates_ith(first_logit_idx);
        let mut first_logits_vec = vec![-f32::INFINITY; self.model.n_vocab() as usize];
        for candidate in first_candidates {
            let token_id = candidate.id().0 as usize;
            if token_id < first_logits_vec.len() {
                first_logits_vec[token_id] = candidate.logit();
            }
        }

        let mut next_token_id = sample_greedy(&first_logits_vec, preset, &mut rng_seed);

        let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();

        // Single linear autoregressive generation loop
        loop {
            if next_token_id == preset.eos_token as i32 {
                break;
            }

            let next_token = LlamaToken(next_token_id);
            
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

            generated_tokens.push(next_token);

            // Wipe prompt leftovers and stage exactly one token at index 0
            batch.clear();
            batch.add(next_token, current_pos, &[0], true)?;
            
            if let Err(e) = ctx.decode(&mut batch) {
                eprintln!("\n[ERROR-LOOP] Decode failed at pos {}: {:?}", current_pos, e);
                break;
            }

            current_pos += 1;

            // Batch size is strictly 1, so the new logit is always at index 0
            let candidates = ctx.candidates_ith(0);
            let mut logits_vec = vec![-f32::INFINITY; self.model.n_vocab() as usize];
            for candidate in candidates {
                let token_id = candidate.id().0 as usize;
                if token_id < logits_vec.len() {
                    logits_vec[token_id] = candidate.logit();
                }
            }

            next_token_id = sample_greedy(&logits_vec, preset, &mut rng_seed);
        }

        let clean_final_text = prev_text.replace("<br>", "\n");
        session.add_message("assistant", clean_final_text.trim());
        session.pos_offset = current_pos as usize;

        Ok(())
    }

    // Allocate persistent execution context space inside GPU VRAM
    pub fn new_context(&self) -> Result<llama_cpp_2::context::LlamaContext<'_>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_batch(2048)
            .with_n_ctx(std::num::NonZeroU32::new(4096));

        let ctx = self.model.new_context(&self.backend, ctx_params)
            .context("Failed to instantiate isolated execution context space")?;
        Ok(ctx)
    }
}