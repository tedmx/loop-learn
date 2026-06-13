use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::SystemTime;
use anyhow::Result;
use candle_transformers::models::bert::{BertModel, Config};

// Import necessary items from Candle ecosystem for vector operations
use candle_core::{Tensor, Device, DType};
use candle_nn::VarBuilder;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatSession {
    pub messages: Vec<ChatMessage>,
}

// Structure to store single indexed knowledge units
pub struct KnowledgeDocument {
    pub text: String,
    pub embedding: Tensor,
}

// Unified core vector registry subsystem
pub struct VectorRegistry {
    pub documents: Vec<KnowledgeDocument>,
    pub bert: BertModel,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
}

#[derive(Serialize, Deserialize)]
struct CachedDocument {
    text: String,
    embedding_data: Vec<f32>,
}

impl VectorRegistry {
    pub fn bootstrap(
        knowledge_path: &Path,
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        let config_file = File::open(config_path)?;
        let config: Config = serde_json::from_reader(config_file)?;

        let hidden_size = config.hidden_size;
        
        let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(model_path)? };
        let vb = VarBuilder::from_backend(Box::new(safetensors), DType::F32, device.clone());
        let bert = BertModel::load(vb, &config)?;
        
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to initialize target embeddings tokenizer structure: {}", e))?;

        let mut documents = Vec::new();

        let cache_path = Path::new("storage/embeddings.json");
        let mtime_path = Path::new("storage/knowledge.mtime");

        let need_reindex = if let (true, true) = (cache_path.exists(), knowledge_path.exists()) {
            let current_mtime = std::fs::metadata(knowledge_path)?.modified()?;
            let stored_mtime = if mtime_path.exists() {
                let content = std::fs::read_to_string(mtime_path)?;
                let secs = content.parse::<u64>().unwrap_or(0);
                SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)
            } else {
                SystemTime::UNIX_EPOCH
            };
            current_mtime > stored_mtime
        } else {
            false
        };

        if cache_path.exists() && !need_reindex {
            println!("Loading pre-computed vector representations from local cache...");
            let cache_file = File::open(cache_path)?;
            let cached_docs: Vec<CachedDocument> = serde_json::from_reader(cache_file)?;
            
            for doc in cached_docs {
                if doc.embedding_data.len() != hidden_size {
                    anyhow::bail!(
                        "Embedding dimension mismatch: cached {} vs model {}",
                        doc.embedding_data.len(),
                        hidden_size
                    );
                }
                let tensor = Tensor::new(doc.embedding_data.as_slice(), device)?;
                documents.push(KnowledgeDocument {
                    text: doc.text,
                    embedding: tensor,
                });
            }
        } else if knowledge_path.exists() {
            println!("Indexing targets located. Constructing vector representations...");

            // Read the entire file content into a unified string buffer
            let content = std::fs::read_to_string(knowledge_path)?;

            let paragraphs: Vec<String> = content
                .split("\n\n")
                .map(|p| {
                    // Replace all single internal newlines with ordinary spaces, then trim
                    p.replace(['\n', '\r'], " ").trim().to_string()
                })
                .filter(|p| !p.is_empty() && p.len() > 10)
                .collect();

            let chunk_size = 200; 
            let overlap = 40;

            // Slide the window across the words array
            for paragraph in paragraphs {
                let words: Vec<&str> = paragraph.split_whitespace().collect();

                // Step 2: If the paragraph fits into a single chunk, index it directly
                if words.len() <= chunk_size {
                    let chunk_text = words.join(" ");
                    
                    let tokens = tokenizer.encode(chunk_text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenizer sequence mapping failure: {}", e))?;
                    
                    let token_ids = tokens.get_ids();
                    let input_ids = Tensor::new(token_ids, device)?.unsqueeze(0)?;
                    let token_type_ids = Tensor::zeros_like(&input_ids)?;

                    let embeddings = bert.forward(&input_ids, &token_type_ids, None)?;
                    let doc_embedding = embeddings.mean_keepdim(1)?.squeeze(1)?.squeeze(0)?;
                    let normalized_doc_embedding = Self::l2_normalize(&doc_embedding)?;

                    documents.push(KnowledgeDocument {
                        text: chunk_text,
                        embedding: normalized_doc_embedding,
                    });
                } else {
                    // Step 3: If the paragraph is huge, process it via sliding window
                    let mut start_idx = 0;
                    while start_idx < words.len() {
                        let end_idx = std::cmp::min(start_idx + chunk_size, words.len());
                        let chunk_text = words[start_idx..end_idx].join(" ");
                        
                        let tokens = tokenizer.encode(chunk_text.as_str(), true)
                            .map_err(|e| anyhow::anyhow!("Tokenizer sequence mapping failure: {}", e))?;
                        
                        let token_ids = tokens.get_ids();
                        let input_ids = Tensor::new(token_ids, device)?.unsqueeze(0)?;
                        let token_type_ids = Tensor::zeros_like(&input_ids)?;

                        let embeddings = bert.forward(&input_ids, &token_type_ids, None)?;
                        let doc_embedding = embeddings.mean_keepdim(1)?.squeeze(1)?.squeeze(0)?;
                        let normalized_doc_embedding = Self::l2_normalize(&doc_embedding)?;

                        documents.push(KnowledgeDocument {
                            text: chunk_text,
                            embedding: normalized_doc_embedding,
                        });

                        start_idx += chunk_size - overlap;

                        if end_idx >= words.len() {
                            break;
                        }
                    }
                }
            }

            let cache_to_save: Vec<CachedDocument> = documents
                .iter()
                .map(|d| {
                    let data = d.embedding.to_vec1::<f32>().unwrap_or_default();
                    CachedDocument { text: d.text.clone(), embedding_data: data }
                })
                .collect();
            
            if let Some(parent) = cache_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut file = File::create(cache_path)?;
            let json_bytes = serde_json::to_vec_pretty(&cache_to_save)?;
            file.write_all(&json_bytes)?;
            println!("Vector cache successfully flushed to disk.");

            let current_mtime = std::fs::metadata(knowledge_path)?.modified()?;
            let mtime_secs = current_mtime.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();
            std::fs::write(mtime_path, mtime_secs.to_string())?;
        } else {
            println!("Warning: Database resource path missing. Standby for empty execution context.");
        }

        Ok(Self { 
            documents, 
            bert, 
            tokenizer, 
            device: device.clone(),
        })
    }

    pub fn find_relevant_context(&mut self, query: &str) -> Result<Option<String>> {
        println!("\n=== [STORAGE] SEARCHING RELEVANT CONTEXT IN KNOWLEDGE BASE ===");
        println!("Query payload: '{}'", query);

        if self.documents.is_empty() {
            println!("Knowledge base is empty. Skipping vector search.");
            return Ok(None);
        }

        // Tokenize incoming query for the BERT model
        let tokens = self.tokenizer.encode(query, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let token_ids = tokens.get_ids();

        println!("[DEBUG] VectorRegistry self.device is: {:?}", self.device);

        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::zeros_like(&input_ids)?;

        println!("[DEBUG] input_ids device: {:?}, shape: {:?}", input_ids.device(), input_ids.shape());
        println!("[DEBUG] token_type_ids device: {:?}, shape: {:?}", token_type_ids.device(), token_type_ids.shape());

        // Extract query embedding layer from BERT forward pass
        let embeddings = self.bert.forward(&input_ids, &token_type_ids, None)?;
        let query_embedding = embeddings.mean_keepdim(1)?.squeeze(1)?.squeeze(0)?;

        println!("[DEBUG] query_embedding device: {:?}, shape: {:?}", query_embedding.device(), query_embedding.shape());

        // Pre-normalize query vector to unit length outside the core loop
        let normalized_query = Self::l2_normalize(&query_embedding)?;

        println!("[DEBUG] normalized_query device: {:?}, shape: {:?}", normalized_query.device(), normalized_query.shape());

        let mut best_score = -1.0f32;
        let mut best_text = Option::<String>::None;

        if let Some(first_doc) = self.documents.first() {
            println!("[DEBUG] Sample document embedding device: {:?}, shape: {:?}", first_doc.embedding.device(), first_doc.embedding.shape());
        }

        println!("[DEBUG] Starting inner distance calculation loop across {} documents...", self.documents.len());

        // Iterate through stored documents to find the highest cosine similarity
        for doc in &self.documents {
            // Since both vectors are L2-normalized, their dot product 
            // is mathematically equal to the exact Cosine Similarity metric.
            let score = (&doc.embedding * &normalized_query)?
                .sum_all()?
                .to_scalar::<f32>()?;

            if score > best_score {
                best_score = score;
                best_text = Some(doc.text.clone());
            }
        }

        println!("Closest knowledge chunk located with similarity score: {:.4}", best_score);
        
        // Match against a safety threshold to prevent injecting irrelevant data
        if best_score > 0.60 {
            if best_text.is_some() {
                println!("Context successfully extracted and verified for LLM context injection.");
            }
            Ok(best_text)
        } else {
            println!("Similarity score below threshold. No relevant context injected.");
            Ok(None)
        }
    }

    fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
        // Calculate the square root of the sum of all squared elements
        let squared_sum = tensor.sqr()?.sum_all()?;
        let norm = squared_sum.sqrt()?;
        
        // Use a tiny epsilon to prevent division by zero on empty or zero vectors
        let epsilon = Tensor::new(1e-12f32, tensor.device())?;
        let safe_norm = norm.maximum(&epsilon)?;
        
        let normalized = tensor.broadcast_div(&safe_norm)?;
        Ok(normalized)
    }
}

impl ChatSession {
    // Load session from disk. If file missing, initialize an empty session state.
    pub fn load_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        
        if !path_ref.exists() {
            // Create parent storage directory directory if it does not exist
            if let Some(parent) = path_ref.parent() {
                std::fs::create_dir_all(parent)?;
            }
            return Ok(Self {
                messages: Vec::new(),
            });
        }

        let mut file = File::open(path_ref)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let session: ChatSession = serde_json::from_str(&contents)?;
        Ok(session)
    }

    // Persist current chat history session structure to disk
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json_bytes = serde_json::to_vec_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(&json_bytes)?;
        Ok(())
    }

    // Push new message into session lifecycle tracking array
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
        
        // Keep context window compact by dropping oldest tokens
        if self.messages.len() > 20 {
            self.messages.drain(0..self.messages.len() - 20);
        };
    }
}