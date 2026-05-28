use anyhow::Result;
use std::fs::File;
use candle_core::Device;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use candle_transformers::models::quantized_qwen2::ModelWeights as QuantizedModel;
use crate::loader::ModelFiles;
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;

pub enum EngineModel {
    Standard(ModelForCausalLM),
    Quantized(QuantizedModel),
}

pub struct InferenceEngine {
    pub model: EngineModel,
    pub device: Device,
    pub dtype: candle_core::DType,
}

impl InferenceEngine {
    pub fn new(files: &ModelFiles, device: &Device, dtype: candle_core::DType) -> Result<Self> {
        println!("Загрузка конфигурации модели...");
        let config_file = File::open(&files.config)?;
        let config: Config = serde_json::from_reader(&config_file)?;

        let safetensors = unsafe { 
            candle_core::safetensors::MmapedSafetensors::new(&files.weights)? 
        };

        println!("Using Data Type {:?}", dtype);
        let vb = candle_nn::VarBuilder::from_backend(
            Box::new(safetensors),
            dtype,
            device.clone()
        );

        println!("Инициализация architecture Qwen2...");
        let model = ModelForCausalLM::new(&config, vb)?;
        println!("Модель успешно развернута в памяти и готова к генерации.");

        Ok(Self {
            model: EngineModel::Standard(model),
            device: device.clone(),
            dtype,
        })
    }

    pub fn new_gguf(gguf_path: &std::path::Path, device: &Device) -> Result<Self> {
        println!("Opening GGUF model file...");
        let mut file = std::fs::File::open(gguf_path)?;
        
        // Read GGUF content metadata as explicitly demanded by the compiler signature
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        
        println!("Initializing Quantized Qwen2 architecture via Llama ModelWeights constructor...");
        // Supplying exactly 3 arguments: content, &mut file, and target compute device
        let model = QuantizedModel::from_gguf(content, &mut file, device)?;
        println!("Quantized GGUF model loaded successfully into VRAM.");
    
        Ok(Self {
            model: EngineModel::Quantized(model),
            device: device.clone(),
            dtype: candle_core::DType::F32,
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
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?
            .get_ids()
            .to_vec();

        let mut current_pos = 0;
        let model_dtype = self.dtype;

        // Process prompt: both F32 and BF16 can securely execute batch forward operations
        let input_tensor = candle_core::Tensor::new(new_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = match &mut self.model {
            EngineModel::Standard(m) => m.forward(&input_tensor, current_pos)?,
            EngineModel::Quantized(m) => m.forward(&input_tensor, current_pos)?.unsqueeze(0)?,
        };
        current_pos += new_tokens.len();

        let last_logits = if logits.rank() == 3 {
            logits.get(0)?.get(logits.dim(1)? - 1)?
        } else if logits.rank() == 2 {
            logits.get(0)?
        } else {
            logits
        }.contiguous()?;

        // Cast low-precision weights to F32 during processing to eliminate sampling artifacts
        let last_logits = if model_dtype == candle_core::DType::BF16 {
            last_logits.to_dtype(candle_core::DType::F32)?
        } else {
            last_logits
        };

        let current_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut logits_processor = LogitsProcessor::new(current_seed, None, None);
        let eos_token = 151645_u32;
        let mut tokens_queue = Vec::new();
        let mut prev_text = String::new();

        println!("\nAI response: ");

        let mut next_token = logits_processor.sample(&last_logits)?;
        println!("[DEBUG] Первый выбранный токен ID: {}", next_token);

        for _ in 0..1024 {
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

            let logits = match &mut self.model {
                EngineModel::Standard(m) => m.forward(&input, gen_pos)?,
                EngineModel::Quantized(m) => m.forward(&input, gen_pos)?.unsqueeze(0)?,
            };

            let logits = if logits.rank() == 3 {
                logits.get(0)?.get(logits.dim(1)? - 1)?
            } else if logits.rank() == 2 {
                logits.get(0)?
            } else {
                logits
            }.contiguous()?;
            
            // Safe execution branch isolating standard pipeline from low-memory paths
            let logits = match &self.model {
                EngineModel::Standard(_) => {
                    if model_dtype == candle_core::DType::BF16 {
                        logits.to_dtype(candle_core::DType::F32)?
                    } else {
                        logits
                    }
                },
                EngineModel::Quantized(_) => logits, // GGUF logits are native F32
            };

            next_token = logits_processor.sample(&logits)?;

            if next_token == 151643 {
                break;
            }
        }

        println!();

        session.add_message("assistant", prev_text.trim());
        session.pos_offset = current_pos;

        Ok(())
    }
}
