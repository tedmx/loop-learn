use candle_core::Device;
use anyhow::Result;
use std::path::Path;

// Декларируем изолированные компоненты нашего замкнутого цикла (Feedback Loop)
mod loader;    // Загрузка и кэширование весов
mod inference; // Быстрый локальный инференс
mod train;     // Фоновое дообучение
mod storage;   // Память и буфер воспроизведения

use storage::ChatSession;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("=== Инициализация системы loop-learn ===");
    println!("Выбранное устройство вычислений: {:?}", device);
    
    // Вызываем наш загрузчик. Оператор `?` вернет ошибку, если что-то пойдет не так (например, пропадет интернет)
    let env_files = loader::ModelFiles::download()?;

    let mut engine = inference::InferenceEngine::new(&env_files, &device)?;

    let args: Vec<String> = std::env::args().collect();
    
    // Ищем позицию флага -p и берем значение за ним
    let prompt_arg = if let Some(pos) = args.iter().position(|x| x == "-p") {
        args.get(pos + 1).map(|s| s.as_str())
    } else {
        None
    };

    if let Some(prompt_str) = prompt_arg {
        let history_path = Path::new("storage/chat_history.json");
        let mut session = ChatSession::load_or_create(history_path)?;

        let context_info = crate::storage::find_relevant_context(prompt_str);

       let execution_prompt = if let Some(ref facts) = context_info {
            println!("Найден релевантный контекст в локальной базе знаний. Инжектируем факты...");
            format!(
                "Контекст из базы знаний:\n{}\n\nИспользуя этот контекст, ответь на вопрос: {}", 
                facts, 
                prompt_str
            )
        } else {
            prompt_str.to_string()
        };

        session.add_message("user", prompt_str);

        // Запускаем движок генерации
        if let Err(e) = engine.generate(&mut session, &execution_prompt, &env_files.tokenizer) {
            println!("Ошибка генерации: {:?}", e);
        } else {
            // Сбрасываем сдвиг позиций и сохраняем чистую историю диалога
            session.pos_offset = 0;
            session.save(history_path)?;
        }
    }

    Ok(())
}
