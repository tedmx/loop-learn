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

    // Авторегрессионная генерация ответа
    pub fn generate(&mut self, prompt: &str, tokenizer_path: &std::path::Path) -> Result<()> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Ошибка загрузки токенизатора: {}", e))?;

        // Шаблон для TinyLlama
        let formatted_prompt = format!(
            "<|im_start|>system\nТы полезный AI-ассистент.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );

        let tokens = tokenizer.encode(formatted_prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Ошибка кодирования: {}", e))?;
        let mut tokens_queue = tokens.get_ids().to_vec();

        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));
        let eos_token = 151645_u32;

        let mut start_pos = 0;

        println!("\nОтвет ИИ: ");

        // 1. Сначала «прогреваем» модель входным промптом, скармливая строго по ОДНОМУ токену.
        // Это заставит модель корректно заполнить KV-кэш без падения на длинных масках.
        for i in 0..tokens_queue.len() {
            let single_token = vec![tokens_queue[i]];
            let input = candle_core::Tensor::new(single_token.as_slice(), &self.device)?
                .unsqueeze(0)?; // [1, 1]

            let logits = self.model.forward(&input, start_pos)?;
            let logits = if logits.rank() == 3 {
                logits.get(0)?.get(logits.dim(1)? - 1)?
            } else if logits.rank() == 2 {
                logits.get(0)?
            } else {
                logits
            }.contiguous()?;
            
            // Если это самый последний токен промпта, нам нужно вытащить его логиты для генерации
            if i == tokens_queue.len() - 1 {
                let logits = logits.squeeze(0)?.contiguous()?;
                
                let mut next_token = logits_processor.sample(&logits)?;

                let mut prev_text = tokenizer.decode(&tokens_queue, true)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                
                // Крутим цикл генерации новых токенов
                for _ in 0..200 {
                    if next_token == eos_token {
                        break;
                    }

                    tokens_queue.push(next_token);
                    start_pos += 1;

                    if let Ok(text) = tokenizer.decode(&tokens_queue, true) {
                        if text.len() > prev_text.len() {
                            let new_part = &text[prev_text.len()..];
                            print!("{}", new_part);
                            std::io::Write::flush(&mut std::io::stdout())?;
                        }
                        prev_text = text;
                    }

                    // Передаем ОДИН только что сгенерированный токен
                    let input = candle_core::Tensor::new(&[next_token], &self.device)?
                        .unsqueeze(0)?;
                    
                let logits = self.model.forward(&input, start_pos)?;
                let logits = if logits.rank() == 3 {
                    logits.get(0)?.get(logits.dim(1)? - 1)?
                } else if logits.rank() == 2 {
                    logits.get(0)?
                } else {
                    logits
                }.contiguous()?;
                
                    next_token = logits_processor.sample(&logits)?;
                }
            } else {
                // Для всех промежуточных токенов промпта просто инкрементируем позицию в кэше
                start_pos += 1;
            }
        }
        println!();
        Ok(())
    }

}
