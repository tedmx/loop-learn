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
    pub fn generate(&mut self, session: &mut super::storage::ChatSession, prompt: &str, tokenizer_path: &std::path::Path) -> Result<()> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Ошибка загрузки токенизатора: {}", e))?;

        // --- ФАЗА 1: ПРОГРЕВ КЭША (HYDRATION) ---
        let history_chatml = session.to_chatml();println!("\n=== [DEBUG] СЫРОЙ КОНТЕКСТ ДЛЯ МОДЕЛИ ===");
        println!("{}", history_chatml);
        println!("=========================================\n");

        let history_tokens = tokenizer.encode(history_chatml.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Ошибка кодирования истории: {}", e))?
            .get_ids()
            .to_vec();

        let mut current_pos = 0;

        if !history_tokens.is_empty() {
            print!("Восстановление контекста памяти... ");
            std::io::Write::flush(&mut std::io::stdout())?;

            // Создаем ОДИН тензор формы [1, seq_len] из всей истории сразу
            let input = candle_core::Tensor::new(history_tokens.as_slice(), &self.device)?
                .unsqueeze(0)?;
            
            // Прокачиваем всю историю за один шаг, стартуя с позиции 0
            let _logits = self.model.forward(&input, 0)?;
            
            current_pos = history_tokens.len();
            println!("готово (обработано {} токенов).", current_pos);
        }

        session.pos_offset = current_pos;

        let new_prompt_chatml = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);
        let new_tokens = tokenizer.encode(new_prompt_chatml.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Ошибка кодирования нового промпта: {}", e))?
            .get_ids()
            .to_vec();

        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));
        let eos_token = 151645_u32;
        let mut tokens_queue = Vec::new();
        let mut generated_text = String::new();

        println!("\nОтвет ИИ: ");

        // Прокачиваем токены нового промпта
        for (i, &token) in new_tokens.iter().enumerate() {
            let input = candle_core::Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, current_pos)?;

            // Если это последний токен нового промпта, берем логиты и сэмплируем первый токен ответа
            if i == new_tokens.len() - 1 {
                let logits = if logits.rank() == 3 {
                    logits.get(0)?.get(logits.dim(1)? - 1)?
                } else if logits.rank() == 2 {
                    logits.get(0)?
                } else {
                    logits
                }.contiguous()?;

                current_pos += 1;
                let mut next_token = logits_processor.sample(&logits)?;
                let mut prev_text = String::new();

                // Цикл генерации текста ответа
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
                generated_text = prev_text;
            } else {
                current_pos += 1;
            }
        }

        println!();

        session.add_message("assistant", generated_text.trim());
        
        // Пересчитываем итоговый pos_offset на основе обновленной истории
        let final_chatml = session.to_chatml();
        let final_tokens = tokenizer.encode(final_chatml.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Ошибка финального подсчета токенов: {}", e))?
            .get_ids()
            .to_vec();
            
        session.pos_offset = final_tokens.len();

        Ok(())
    }
}
