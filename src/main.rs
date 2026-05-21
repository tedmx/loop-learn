use candle_core::Device;
use anyhow::Result;

// Декларируем изолированные компоненты нашего замкнутого цикла (Feedback Loop)
mod loader;    // Загрузка и кэширование весов
mod inference; // Быстрый локальный инференс
mod train;     // Фоновое дообучение
mod storage;   // Память и буфер воспроизведения

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("=== Инициализация системы loop-learn ===");
    println!("Выбранное устройство вычислений: {:?}", device);
    
    // Вызываем наш загрузчик. Оператор `?` вернет ошибку, если что-то пойдет не так (например, пропадет интернет)
    let env_files = loader::ModelFiles::download()?;

    let mut engine = inference::InferenceEngine::new(&env_files, &device)?;
    
    // Запускаем инференс с бытовым системным промптом
    engine.generate(
        "Что такое токен в контексте нейронной сети?", 
        &env_files.tokenizer
    )?;
    
    Ok(())
}
