use candle_core::{Tensor, Device, DType};
use candle_nn::{VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use anyhow::Result;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
  pub role: String,
  pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatSession {
  pub messages: Vec<ChatMessage>,
  pub pos_offset: usize,
}

impl ChatSession {
    // Загрузка сессии с диска. Если файла нет — создаем пустую сессию
    pub fn load_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        
        if !path_ref.exists() {
            // Если папки storage/ нет, создаем ее
            if let Some(parent) = path_ref.parent() {
                std::fs::create_dir_all(parent)?;
            }
            return Ok(Self {
                messages: Vec::new(),
                pos_offset: 0,
            });
        }

        let mut file = File::open(path_ref)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let session: ChatSession = serde_json::from_str(&contents)?;
        Ok(session)
    }

    // Сохранение текущего состояния сессии на диск
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json_bytes = serde_json::to_vec_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(&json_bytes)?;
        Ok(())
    }

    // Добавление новой реплики в историю
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
        // Держим историю компактной, удаляя старые сообщения, если превышен лимит
        if self.messages.len() > 20 {
            self.messages.drain(0..self.messages.len() - 20);
        };
    }
}

// Структура для хранения проиндексированного блока знаний
pub struct KnowledgeDocument {
    pub text: String,
    pub embedding: Tensor,
}

pub struct VectorRegistry {
    pub documents: Vec<KnowledgeDocument>,
    pub bert: BertModel,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
}

impl VectorRegistry {
    // Индексируем текстовый файл, превращая каждый блок в вектор
    pub fn bootstrap(
        knowledge_path: &Path,
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let config_file = File::open(config_path)?;
        let config: Config = serde_json::from_reader(&config_file)?;
        
        let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(model_path)? };
        let vb = VarBuilder::from_backend(Box::new(safetensors), DType::F32, device.clone());
        let bert = BertModel::load(vb, &config)?;
        
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Ошибка загрузки токенизатора эмбеддингов: {}", e))?;

        let mut documents = Vec::new();

        if knowledge_path.exists() {
            let file = File::open(knowledge_path)?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;

            let mut current_section = String::new();

            for line in reader.lines() {
                let line = line?.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                // Если строка — это заголовок секции, запоминаем её и НЕ индексируем отдельно
                if line.starts_with("===") && line.ends_with("===") {
                    current_section = line.replace("===", "").trim().to_string();
                    continue;
                }

                // К каждому контентному абзацу подмешиваем имя его секции для идеального векторного поиска
                let full_chunk_text = if !current_section.is_empty() {
                    format!("[Section: {}] {}", current_section, line)
                } else {
                    line
                };

                // Генерируем эмбеддинг для склеенного чанка
                let tokens = tokenizer.encode(full_chunk_text.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
                let token_ids = tokens.get_ids();
                let input_ids = Tensor::new(token_ids, &device)?.unsqueeze(0)?;
                let token_type_ids = Tensor::zeros_like(&input_ids)?;

                let embeddings = bert.forward(&input_ids, &token_type_ids, None)?;
                let embedding = embeddings.mean_keepdim(1)?.squeeze(1)?.squeeze(0)?;

                // Сохраняем в документ именно полный текст с фактами
                documents.push(KnowledgeDocument {
                    text: full_chunk_text,
                    embedding,
                });
            }
        }

        Ok(Self { 
            documents, 
            bert, 
            tokenizer, 
            device: device.clone() 
        })
    }


    // Высокоуровневый семантический поиск: принимает строку текста и возвращает контекст
    pub fn find_relevant_context(&mut self, query: &str) -> Result<Option<String>> {
        // 1. Получаем эмбеддинг для запроса пользователя
        let tokens = self.tokenizer.encode(query, true)
            .map_err(|e| anyhow::anyhow!("Query tokenizer error: {}", e))?;
        let token_ids = tokens.get_ids();
        
        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::zeros_like(&input_ids)?;
        let embeddings = self.bert.forward(&input_ids, &token_type_ids, None)?;
        let query_vector = embeddings.mean_keepdim(1)?.squeeze(1)?.squeeze(0)?;

        // 2. Считаем косинусное сходство со всеми документами
        let mut best_text = None;
        let mut max_similarity = -1.0;

        println!("\n=== [DEBUG] СТАРТ ПОИСКА ПО БАЗЕ ЗНАНИЙ ===");
        println!("Запрос: '{}'", query);

        for (idx, doc) in self.documents.iter().enumerate() {
            let dot_product = (&doc.embedding * &query_vector)?.sum_all()?.to_scalar::<f32>()?;
            let norm_a = doc.embedding.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let norm_b = query_vector.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            
            let denominator = norm_a * norm_b;
            let similarity = if denominator > 1e-6 {
                dot_product / denominator
            } else {
                0.0
            };

            // Выводим имя блока (первые 35 символов) и его честный скор
            let short_title: String = doc.text.chars().take(35).collect();
            println!("  -> Блок №{} [{}...]: similarity = {:.4}", idx, short_title.replace('\n', " "), similarity);

            if similarity > max_similarity {
                max_similarity = similarity;
                best_text = Some(doc.text.clone());
            }
        }

        println!("-> Vector search completed. Max similarity score: {:.4}", max_similarity);

        if max_similarity > 0.45 {
            Ok(best_text)
        } else {
            Ok(None)
        }
    }
}
