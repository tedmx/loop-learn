use anyhow::Result;
use std::fs::File;
use candle_core::Device;
use candle_transformers::models::llama::{Config, Llama as Model, Cache, LlamaConfig};
use crate::loader::ModelFiles;
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;

pub struct InferenceEngine {
    pub model: Model,
    pub config: Config,
    pub device: Device,
}

impl InferenceEngine {
    pub fn new(files: &ModelFiles, device: &Device) -> Result<Self> {
        println!("Загрузка конфигурации модели...");
        let config_file = File::open(&files.config)?;
        
        let llama_config: LlamaConfig = serde_json::from_reader(config_file)?;
        let config = llama_config.into_config(false);

        let safetensors = unsafe { 
            candle_core::safetensors::MmapedSafetensors::new(&files.weights)? 
        };
        
        let vb = candle_nn::VarBuilder::from_backend(
            Box::new(safetensors),
            candle_core::DType::F16,
            device.clone()
        );

        println!("Инициализация architecture Llama...");
        // ИСПРАВЛЕНИЕ: Используем load вместо new, как просит компилятор
        let model = Model::load(vb, &config)?;

        println!("Модель успешно развернута в памяти и готова к генерации.");

        Ok(Self {
            model,
            config,
            device: device.clone(),
        })
    }

    // Авторегрессионная генерация ответа
    pub fn generate(&mut self, prompt: &str, tokenizer_path: &std::path::Path) -> Result<()> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Ошибка загрузки токенизатора: {}", e))?;

        // Шаблон для TinyLlama
        let chat_prompt = format!("<|user|>\n{}</s>\n<|assistant|>\n", prompt);

        let tokens = tokenizer.encode(chat_prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Ошибка кодирования: {}", e))?;
        let mut tokens_queue = tokens.get_ids().to_vec();

        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));
        let eos_token = tokenizer.token_to_id("</s>").unwrap_or(2);

        let mut cache = Cache::new(true, candle_core::DType::F16, &self.config, &self.device)?;
        let mut start_pos = 0;

        println!("\nОтвет ИИ: ");

        // 1. Сначала «прогреваем» модель входным промптом, скармливая строго по ОДНОМУ токену.
        // Это заставит модель корректно заполнить KV-кэш без падения на длинных масках.
        for i in 0..tokens_queue.len() {
            let single_token = vec![tokens_queue[i]];
            let input = candle_core::Tensor::new(single_token.as_slice(), &self.device)?
                .unsqueeze(0)?; // [1, 1]

            let logits = self.model.forward(&input, start_pos, &mut cache)?;
            
            // Если это самый последний токен промпта, нам нужно вытащить его логиты для генерации
            if i == tokens_queue.len() - 1 {
                let logits = logits.squeeze(0)?.contiguous()?;
                
                let mut next_token = logits_processor.sample(&logits)?;

                let mut prev_text = tokenizer.decode(&tokens_queue, true)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                
                // Крутим цикл генерации новых токенов
                for _ in 0..100 {
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
                    
                    let logits = self.model.forward(&input, start_pos, &mut cache)?;
                    let logits = logits.squeeze(0)?.contiguous()?;
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
