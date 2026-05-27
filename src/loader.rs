use anyhow::Result;
use std::path::PathBuf;
use hf_hub::api::sync::ApiBuilder;

// Используем легковесную, точную и отлично знающую русский язык модель
const REPO_ID: &str = "Qwen/Qwen2.5-1.5B-Instruct";
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

        let cache = hf_hub::Cache::default();
        let repo_cache = cache.repo(hf_hub::Repo::new(REPO_ID.to_string(), hf_hub::RepoType::Model));

        let tokenizer_path = repo_cache.get(TOKENIZER_FILE);
        let config_path = repo_cache.get(CONFIG_FILE);
        let weights_path = repo_cache.get("model.safetensors");

        if let (Some(tokenizer), Some(config), Some(weights)) = (tokenizer_path, config_path, weights_path) {
        println!("Все необходимые файлы модели верифицированы и готовы к работе.");
        return Ok(Self {
            tokenizer,
            config,
            weights,
        });
        };

        println!("Файлы модели не найдены локально. Подключение к Hugging Face для скачивания...");
        let api = ApiBuilder::new()
        .with_progress(true)
        .build()?;
        let repo = api.model(REPO_ID.to_string());

        let tokenizer = repo.get(TOKENIZER_FILE)?;
        let config = repo.get(CONFIG_FILE)?;
        let weights = repo.get("model.safetensors")?;

        println!("Все необходимые файлы модели успешно загружены.");
        return Ok(Self {
        tokenizer,
        config,
        weights,
        });
    }
}

// Структура для путей к файлам модели эмбеддингов
pub struct EmbeddingFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
}

impl EmbeddingFiles {
    // Автоматическая загрузка или верификация локального кэша для BGE-модели
    pub fn download_or_get() -> anyhow::Result<Self> {
        let repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
        println!("Проверка локального кэша для мультиязычной модели эмбеддингов: {}", repo_id);

        let cache = hf_hub::Cache::default();
        let repo_cache = cache.repo(hf_hub::Repo::new(repo_id.to_string(), hf_hub::RepoType::Model));

        let config_path = repo_cache.get("config.json");
        let weights_path = repo_cache.get("model.safetensors");
        let tokenizer_path = repo_cache.get("tokenizer.json");

        if let (Some(config), Some(weights), Some(tokenizer)) = (config_path, weights_path, tokenizer_path) {
        println!("Все необходимые файлы эмбеддингов верифицированы из локального кэша.");
        return Ok(Self {
            config,
            weights,
            tokenizer,
        });
        };

        println!("Файлы эмбеддингов не найдены локально. Подключение к Hugging Face для скачивания...");
        let api = ApiBuilder::new()
        .with_progress(true)
        .build()?;
        let repo = api.model(repo_id.to_string());

        let config = repo.get("config.json")
        .map_err(|e| anyhow::anyhow!("Не удалось загрузить config.json для эмбеддингов: {}", e))?;
        let weights = repo.get("model.safetensors")
        .map_err(|e| anyhow::anyhow!("Не удалось загрузить model.safetensors для эмбеддингов: {}", e))?;
        let tokenizer = repo.get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Не удалось загрузить tokenizer.json для эмбеддингов: {}", e))?;

        println!("Все необходимые файлы эмбеддингов успешно загружены.");
        return Ok(Self {
        config,
        weights,
        tokenizer,
        });
    }
}