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
    }

    // Сборка всей истории в один ChatML-текст для прогрева кэша
    pub fn to_chatml(&self) -> String {
        let mut result = String::new();

        result.push_str("<|im_start|>system\nТы полезный AI-ассистент. Ты внимательно анализируешь ВСЮ цепочку сообщений ниже и помнишь абсолютно каждый факт из прошлых тем обсуждения.<|im_end|>\n\n");

        for msg in &self.messages {
            result.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        
        result
    }
}

// Функция для поиска наиболее релевантного блока знаний по ключевым словам
pub fn find_relevant_context(query: &str) -> Option<String> {
    let knowledge_path = std::path::Path::new("storage/knowledge.txt");
    if !knowledge_path.exists() {
        return None;
    }

    if let Ok(content) = std::fs::read_to_string(knowledge_path) {
        // Разбираем файл на отдельные смысловые блоки
        let blocks: Vec<&str> = content.split("===").collect();
        let query_lower = query.to_lowercase();
        // Разбиваем запрос на отдельные слова для поиска пересечений
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut best_block = None;
        let mut max_matches = 0;

        for block in blocks {
            let block_lower = block.to_lowercase();
            let mut matches = 0;
            
            for word in &words {
                // Игнорируем слишком короткие слова (предлоги, союзы), чтобы избежать ложных срабатываний
                if word.len() > 3 && block_lower.contains(word) {
                    matches += 1;
                }
            }

            // Если в этом блоке больше совпадений, чем в предыдущих, запоминаем его
            if matches > max_matches {
                max_matches = matches;
                best_block = Some(block.to_string());
            }
        }

        // Если нашли блок, в котором совпало хотя бы одно значимое ключевое слово
        if max_matches > 0 {
            if let Some(mut block_str) = best_block {
                // Очищаем от служебных заголовков, если они остались в начале строки
                if let Some(index) = block_str.find('\n') {
                    if block_str[..index].contains(':') || block_str[..index].trim().is_empty() {
                        block_str = block_str[index..].to_string();
                    }
                }
                return Some(block_str.trim().to_string());
            }
        }
    }

    None
}

