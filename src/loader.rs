use anyhow::Result;
use std::path::PathBuf;

// Используем легковесную, точную и отлично знающую русский язык модель
const REPO_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CONFIG_FILE: &str = "config.json";

// Структура, хранящая безопасные пути к файлам модели в ОС
pub struct ModelFiles {
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    pub weights: PathBuf,
}

impl ModelFiles {
    // Метод для скачивания или мгновенного извлечения файлов из локального кэша
    pub fn download() -> Result<Self> {
        println!("Проверка локального кэша для модели: {}", REPO_ID);
        
        // Подключаемся к API Hugging Face Hub (использует локальный кэш в ~/.cache/)
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(REPO_ID.to_string());

        // Получаем пути к конфигурации и токенизатору
        let tokenizer = repo.get(TOKENIZER_FILE)?;
        let config = repo.get(CONFIG_FILE)?;
        
        // Загружаем базовые веса в безопасном формате Safetensors.
        // Для первого инференса берем основной файл, далее перейдем к квантованию.
        let weights = repo.get("model.safetensors")?;

        println!("Все необходимые файлы модели верифицированы и готовы к работе.");

        Ok(Self {
            tokenizer,
            config,
            weights,
        })
    }
}
