#[link(name = "nccl")]
unsafe extern "C" {}

use anyhow::{Context, Result};
use std::path::Path;
use std::io::Write;

mod loader; 
mod inference;
mod storage;
mod presets;

use storage::ChatSession;
use loader::ModelFiles;
use inference::InferenceEngine;

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

    let preset = crate::presets::PRESETS
        .iter()
        .find(|p| p.name == preset_name)
        .ok_or_else(|| anyhow::anyhow!("CLI Error: Profile '{}' not found. Execute with --list-presets to view choices.", preset_name))?;

    // Extract hardware routing directly from runtime configuration argument or preset
    let cuda_index = if let Some(pos) = args.iter().position(|x| x == "--cuda-index") {
        if pos + 1 < args.len() {
            args[pos + 1].parse::<i32>().unwrap_or(0)
        } else {
            0
        }
    } else if preset.vram_limit == "12 GB" {
        1
    } else {
        0
    };

    let model_files = ModelFiles::download_model_files(preset)
        .context("Failed to prepare GGUF model files")?;
    
    println!("Initializing local hardware execution context...");
    let engine = InferenceEngine::new(&model_files, cuda_index)
        .context("Failed to build unified inference engine lifecycle")?;

    let storage_dir = Path::new("storage/session.json");
    let knowledge_txt = Path::new("storage/knowledge.txt");
    let bert_model_path = Path::new("storage/model.safetensors");
    let bert_config_path = Path::new("storage/config.json");
    let tokenizer_path = Path::new("storage/tokenizer.json");

    let candle_cpu_device = candle_core::Device::Cpu;

    println!("Bootstrapping vector registration registry subsystem...");
    let mut vector_registry = storage::VectorRegistry::bootstrap(
        knowledge_txt,
        bert_model_path,
        bert_config_path,
        tokenizer_path,
        &candle_cpu_device,
    ).context("Failed to bootstrap target VectorRegistry repository state")?;

    let mut session = ChatSession::load_or_create(storage_dir)?;

    // --- Step 3: Extract User Input from CLI Arguments ---
    let prompt_arg = if let Some(pos) = args.iter().position(|x| x == "--prompt") {
        if pos + 1 < args.len() {
            args[pos + 1].as_str()
        } else {
            anyhow::bail!("CLI Error: Missing text payload after --prompt flag");
        }
    } else {
        anyhow::bail!("CLI Error: No prompt provided. Use the --prompt flag to submit your query.");
    };

    let prompt_str = prompt_arg.trim();
    if prompt_str.is_empty() {
        anyhow::bail!("CLI Error: Submitted prompt parameter is completely empty");
    }

    // --- Step 4: Context Extraction (Mocked) ---
    let context_info = vector_registry.find_relevant_context(prompt_str)?;

    let system_instruction = "You are a helpful local AI assistant. If the context contains relevant information regarding the user's question, prioritize using it for an accurate answer. If there is no context or the context lacks information, answer to the best of your own general knowledge.";


    session.add_message("user", prompt_str);

    let mut history_context = String::new();
    for msg in &session.messages {
        history_context.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role,
            msg.content
        ));
    };

    // Assemble the single unified prompt architecture matching ChatML specifications
    let execution_prompt = if let Some(ref facts) = context_info {
        println!("Context extraction found, payload size: {} chars", facts.len());
        format!(
            "<|im_start|>system\n{} Here is the context from the knowledge base:\n{}\n<|im_end|>\n{}<|im_start|>assistant\n",
            system_instruction,
            facts,
            history_context
        )
    } else {
        println!("Context lookup empty, utilizing pure model generation capabilities");
        format!(
            "<|im_start|>system\n{}<|im_end|>\n{}<|im_start|>assistant\n",
            system_instruction,
            history_context
        )
    };

    // --- DEBUG PROMPT INJECTION START ---
    println!("\n[DEBUG] === FULL TEXT SENT TO LLAMA.CPP ===");
    println!("{}", execution_prompt);
    println!("[DEBUG] ===================================\n");
    // --- DEBUG PROMPT INJECTION END ---

    print!("Assistant: ");
    std::io::stdout().flush()?;

    // --- Step 5: Trigger Unified Execution via llama.cpp Core ---
    if let Err(e) = engine.generate(&execution_prompt, &mut session) {
        println!("Generation sequence encountered an error: {:?}", e);
    } else {
        // Persist state updates to disk before application lifecycle termination
        if let Err(e) = session.save(storage_dir) {
            println!("Warning: Failed to flush chat history session to state ledger: {:?}", e);
        }
    }

    Ok(())
}
