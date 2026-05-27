use anyhow::Result;
use std::path::Path;

// Декларируем изолированные компоненты нашего замкнутого цикла (Feedback Loop)
mod loader;    // Загрузка и кэширование весов
mod inference; // Быстрый локальный инференс
mod storage;   // Память и буфер воспроизведения

use storage::ChatSession;

fn main() -> Result<()> {
    let env_files = crate::loader::ModelFiles::download()?; 
    let device = candle_core::Device::new_cuda(0) 
        .unwrap_or(candle_core::Device::Cpu);
    let mut engine = crate::inference::InferenceEngine::new(&env_files, &device)?;

    // Создаем объект токенизатора, который будет доступен во всей функции main
    let tokenizer = tokenizers::Tokenizer::from_file(&env_files.tokenizer)
        .map_err(|e| anyhow::anyhow!(e))?;

    let args: Vec<String> = std::env::args().collect();
    let prompt_arg = if let Some(pos) = args.iter().position(|x| x == "-p") {
        args.get(pos + 1).map(|s| s.as_str())
    } else {
        None
    };

    if let Some(prompt_str) = prompt_arg {
        let history_path = Path::new("storage/chat_history.json");
        let mut session = ChatSession::load_or_create(history_path)?;

        // --- ИЗОЛИРОВАННЫЙ БЛОК ДЛЯ СЕМАНТИЧЕСКОГО ПОИСКА ---
        let context_info = {
            let embed_files = crate::loader::EmbeddingFiles::download_or_get()?;
            let knowledge_path = std::path::Path::new("storage/knowledge.txt");

            println!("Инициализация векторного реестра Базы Знаний...");
            let mut registry = crate::storage::VectorRegistry::bootstrap(
                knowledge_path,
                &embed_files.weights,
                &embed_files.config,
                &embed_files.tokenizer,
                &candle_core::Device::Cpu, // используем тот же девайс, что и у основного движка
            )?;

            let res = registry.find_relevant_context(prompt_str)?;

            res
        };

        let system_instruction = "You are a helpful local AI assistant. If the provided knowledge base context contains information regarding the user's question, prioritize using it for an accurate answer. If there is no context or the context lacks information, answer to the best of your own general knowledge.";

        let execution_prompt = if let Some(ref facts) = context_info {
            println!("Context found, length: {}", facts.len());
            format!(
                "<|im_start|>system\n{} Here is the context from the knowledge base:\n{}\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system_instruction,
                facts,
                prompt_str
            )
        } else {
            println!("Context NOT found, using model's free thinking");
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system_instruction,
                prompt_str
            )
        };

        println!("Текст execution_prompt полностью:\n---\n{}\n---", execution_prompt);
        
        session.add_message("user", prompt_str);

        println!("Передаем управление в engine.generate...");
        // Запускаем движок генерации
        if let Err(e) = engine.generate(&mut session, &execution_prompt, &tokenizer) {
            println!("Ошибка генерации: {:?}", e);
        } else {
            // Сбрасываем сдвиг позиций и сохраняем чистую историю диалога
            session.pos_offset = 0;
            session.save(history_path)?;
        }
    }

    Ok(())
}
