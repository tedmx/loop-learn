use anyhow::Result;
use std::fs::File;
use candle_core::Device;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use candle_transformers::models::quantized_qwen2::ModelWeights as QuantizedQwen;
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use candle_transformers::models::quantized_phi3::ModelWeights as QuantizedPhi3;
use crate::loader::ModelFiles;
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;

pub enum EngineModel {
    Standard(ModelForCausalLM),
    QuantizedQwen(QuantizedQwen),
    QuantizedLlama(QuantizedLlama),
    QuantizedPhi3(QuantizedPhi3),
}

pub struct InferenceEngine {
    pub model: EngineModel,
    pub device: Device,
    pub dtype: candle_core::DType,
    pub eos_token: u32,
    pub template_kind: String,
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
            eos_token: 151645,
            template_kind: "chatml".to_string(),
        })
    }

    pub fn new_gguf(
        gguf_path: &std::path::Path,
        device: &Device,
        architecture: &str,
        eos_token: u32,
        template_kind: &str,
    ) -> Result<Self> {
        println!("Opening GGUF model file for '{}' architecture...", architecture);
        let mut file = File::open(gguf_path)?;
        let mut content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        // Inject Qwen2-compatible metadata fields by projecting existing Phi3 keys
        if architecture == "qwen" && content.metadata.contains_key("phi3.attention.head_count") {
            println!("Translating Phi3 GGUF metadata layout into Qwen2 target space for Phi-4 compatibility...");
            
            if let Some(val) = content.metadata.get("phi3.attention.head_count").cloned() {
                content.metadata.insert("qwen2.attention.head_count".to_string(), val);
            }
            if let Some(val) = content.metadata.get("phi3.attention.head_count_kv").cloned() {
                content.metadata.insert("qwen2.attention.head_count_kv".to_string(), val);
            }
            if let Some(val) = content.metadata.get("phi3.attention.layer_norm_rms_epsilon").cloned() {
                content.metadata.insert("qwen2.attention.layer_norm_rms_epsilon".to_string(), val);
            }
            if let Some(val) = content.metadata.get("phi3.context_length").cloned() {
                content.metadata.insert("qwen2.context_length".to_string(), val);
            }
            if let Some(val) = content.metadata.get("phi3.embedding_length").cloned() {
                content.metadata.insert("qwen2.embedding_length".to_string(), val);
            }
            if let Some(val) = content.metadata.get("phi3.block_count").cloned() {
                content.metadata.insert("qwen2.block_count".to_string(), val);
            } else if let Some(val) = content.metadata.get("general.block_count").cloned() {
                content.metadata.insert("qwen2.block_count".to_string(), val);
            }
            if let Some(val) = content.metadata.get("phi3.rope.dimension_count").cloned() {
                content.metadata.insert("qwen2.rope.dimension_count".to_string(), val);
            }
        }
    
        let model = match architecture {
            "qwen" => {
                println!("Initializing Quantized Qwen2 architecture...");
                EngineModel::QuantizedQwen(QuantizedQwen::from_gguf(content, &mut file, device)?)
            },
            "llama" => {
                println!("Initializing Quantized Llama architecture...");
                EngineModel::QuantizedLlama(QuantizedLlama::from_gguf(content, &mut file, device)?)
            },
            "phi3" => {
                println!("Initializing Quantized Phi3 architecture...");
                // Supplying false for flash_attn, followed by content, reader and device
                EngineModel::QuantizedPhi3(QuantizedPhi3::from_gguf(false, content, &mut file, device)?)
            },
            _ => anyhow::bail!("Unsupported GGUF architecture: {}", architecture),
        };
        println!("Quantized GGUF model weights loaded successfully into VRAM.");

        Ok(Self {
            model,
            device: device.clone(),
            dtype: candle_core::DType::F32,
            eos_token,
            template_kind: template_kind.to_string(),
        })
    }

    pub fn generate(
        &mut self,
        session: &mut super::storage::ChatSession,
        prompt: &str,
        tokenizer: &Tokenizer,
    ) -> Result<()> {
        // Formulate target prompt based on resolved template kind mapping
        let formatted_prompt = match self.template_kind.as_str() {
            "chatml" => format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt),
            "llama3" => format!("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", prompt),
            "phi3" => format!("<|user|>\n{}<|end|>\n<|assistant|>\n", prompt),
            "phi4" => format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt),
            _ => prompt.to_string(),
        };

        let new_tokens = tokenizer.encode(formatted_prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?
            .get_ids()
            .to_vec();

        let mut current_pos = 0;
        let model_dtype = self.dtype;

        // Process prompt: both F32 and BF16 can securely execute batch forward operations
        let input_tensor = candle_core::Tensor::new(new_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = match &mut self.model {
            EngineModel::Standard(m) => {
                m.forward(&input_tensor, current_pos)?
            },
            EngineModel::QuantizedQwen(m) => {
                m.forward(&input_tensor, current_pos)?
            },
            EngineModel::QuantizedLlama(m) => {
                m.forward(&input_tensor, current_pos)?
            },
            EngineModel::QuantizedPhi3(m) => {
                m.forward(&input_tensor, current_pos)?
            },
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
        let mut tokens_queue = Vec::new();
        let mut prev_text = String::new();

        println!("\nAI response: ");

        let mut next_token = logits_processor.sample(&last_logits)?;
        println!("[DEBUG] Первый выбранный токен ID: {}", next_token);

        for _ in 0..1024 {
            if next_token == self.eos_token {
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
                EngineModel::QuantizedQwen(m) => m.forward(&input, gen_pos)?,
                EngineModel::QuantizedLlama(m) => m.forward(&input, gen_pos)?,
                EngineModel::QuantizedPhi3(m) => m.forward(&input, gen_pos)?,
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
                _ => logits, // All GGUF models are evaluated here and produce native F32 logits
            };

            next_token = logits_processor.sample(&logits)?;

            if next_token == self.eos_token {
                break;
            }
        }

        println!();

        session.add_message("assistant", prev_text.trim());
        session.pos_offset = current_pos;

        Ok(())
    }
}
