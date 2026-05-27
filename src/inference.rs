use anyhow::Result;
use std::fs::File;
use candle_core::Device;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use crate::loader::ModelFiles;
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;

pub struct InferenceEngine {
    pub model: ModelForCausalLM,
    pub device: Device,
}

impl InferenceEngine {
    pub fn new(files: &ModelFiles, device: &Device) -> Result<Self> {
        println!("Загрузка конфигурации модели...");
        let config_file = File::open(&files.config)?;
        let config: Config = serde_json::from_reader(&config_file)?;

        let safetensors = unsafe { 
            candle_core::safetensors::MmapedSafetensors::new(&files.weights)? 
        };

        println!("Использование безопасного формата вычислений: F32");
        let vb = candle_nn::VarBuilder::from_backend(
            Box::new(safetensors),
            candle_core::DType::F32,
            device.clone()
        );

        println!("Инициализация architecture Qwen2...");
        let model = ModelForCausalLM::new(&config, vb)?;
        println!("Модель успешно развернута в памяти и готова к генерации.");

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    pub fn generate(
        &mut self,
        session: &mut super::storage::ChatSession,
        prompt: &str,
        tokenizer: &Tokenizer,
    ) -> Result<()> {
        let new_prompt_chatml = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);
        let new_tokens = tokenizer.encode(new_prompt_chatml.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Ошибка токенизации: {}", e))?
        .get_ids()
        .to_vec();

        let mut current_pos = 0;

        // Отправляем весь RAG-контекст одной пачкой напрямую в тензорные ядра GPU
        let input_tensor = candle_core::Tensor::new(new_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, current_pos)?;

        current_pos += new_tokens.len();

        let last_logits = if logits.rank() == 3 {
        logits.get(0)?.get(logits.dim(1)? - 1)?
        } else if logits.rank() == 2 {
        logits.get(0)?
        } else {
        logits
        }.contiguous()?;

        let current_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut logits_processor = LogitsProcessor::new(current_seed, None, None);
        let eos_token = 151645_u32;
        let mut tokens_queue = Vec::new();
        let mut prev_text = String::new();

        println!("\nОтвет ИИ: ");

        let mut next_token = logits_processor.sample(&last_logits)?;

        for _ in 0..512 {
        if next_token == eos_token {
            break;
        }

        tokens_queue.push(next_token);
        let gen_pos = current_pos;
        current_pos += 1;

        if let Ok(text) = tokenizer.decode(&tokens_queue, true) {
            if text.len() > prev_text.len() {
            print!("{}", &text[prev_text.len()..]);
            std::io::Write::flush(&mut std::io::stdout())?;
            }
            prev_text = text;
        }

        let input = candle_core::Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, gen_pos)?;
        let logits = if logits.rank() == 3 {
            logits.get(0)?.get(logits.dim(1)? - 1)?
        } else if logits.rank() == 2 {
            logits.get(0)?
        } else {
            logits
        }.contiguous()?;

        next_token = logits_processor.sample(&logits)?;
        }

        println!();

        session.add_message("assistant", prev_text.trim());
        session.pos_offset = current_pos;

        Ok(())
    }
}
