use anyhow::Result;
use std::path::Path;

mod loader; 
mod inference;
mod storage;
mod presets;

use storage::ChatSession;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Trigger help output if requested
    if args.iter().any(|x| x == "--list-presets") {
        crate::presets::list_presets();
        return Ok(());
    }

    // Resolve targeted configuration profile, fallback to qwen-3b-q4
    let preset_name = if let Some(pos) = args.iter().position(|x| x == "--preset") {
        if pos + 1 < args.len() {
            args[pos + 1].as_str()
        } else {
            anyhow::bail!("CLI Error: Missing configuration profile name after --preset flag");
        }
    } else {
        "qwen-3b-q4"
    };

    // Locate requested configuration within predefined model layout matrix
    let selected_preset = crate::presets::PRESETS
        .iter()
        .find(|p| p.name == preset_name)
        .ok_or_else(|| anyhow::anyhow!("CLI Error: Profile '{}' not found. Execute with --list-presets to view choices.", preset_name))?;

    // Bind execution context to target GPU index (0 for 6GB target, 1 for 12GB runtime)
    let cuda_index = if selected_preset.vram_limit == "12 GB" { 1 } else { 0 };
    let device = candle_core::Device::new_cuda(cuda_index)
        .unwrap_or(candle_core::Device::Cpu);

    let (mut engine, tokenizer_path) = match selected_preset.kind {
        crate::presets::ModelKind::Gguf => {
            let repo = selected_preset.repo_id;
            let filename = selected_preset.filename
                .ok_or_else(|| anyhow::anyhow!("Internal Error: Target GGUF filename payload is empty"))?;
            
            let gguf_path = crate::loader::ModelFiles::download_gguf(repo, filename)?;
            let tok_path = crate::loader::ModelFiles::download_tokenizer_only(selected_preset.tokenizer_repo)?;
            
            // Resolve runtime architecture target from the preset name prefix
            let architecture = if selected_preset.name.starts_with("phi") {
                "phi3"
            } else if selected_preset.name.starts_with("llama") {
                "llama"
            } else {
                "qwen"
            };

            let eng = crate::inference::InferenceEngine::new_gguf(
                &gguf_path, 
                &device, 
                architecture,
                selected_preset.eos_token,
                selected_preset.template_kind,
            )?;

            (eng, tok_path)
        },
        crate::presets::ModelKind::Standard => {
            let env_files = crate::loader::ModelFiles::download()?; 
            let eng = crate::inference::InferenceEngine::new(&env_files, &device, selected_preset.dtype)?;
            (eng, env_files.tokenizer)
        }
    };

    // Instantiate tokenizer instance using isolated local path setup
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!(e))?;

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
                &device,
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
