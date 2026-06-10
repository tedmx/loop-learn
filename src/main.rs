use anyhow::{Context, Result};
use candle_core::Device;
use std::path::Path;
use std::io::Write;

mod loader; 
mod inference;
mod storage;
mod presets;

use storage::ChatSession;
use loader::ModelFiles;
use inference::InferenceEngine;

const SYSTEM_PROMPT: &str = "You are a helpful, concise, and honest local AI assistant. \
When answering, rely primarily on the provided context from the knowledge base if available. \
If the context does not contain enough information to answer the question, use your general knowledge \
but state clearly that the info was not found in the local database. Keep your answers factual and direct.";

fn main() -> Result<()> {
    println!("[DEBUG] v16\n");

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|x| x == "--clear") {
        println!("[INFO] Cleanup triggered. Removing cached session and embeddings...");
        
        let files_to_remove = ["storage/embeddings.json", "storage/session.json"];
        
        for file_path in &files_to_remove {
            let path = std::path::Path::new(file_path);
            if path.exists() {
                if let Err(e) = std::fs::remove_file(path) {
                    println!("[WARN] Failed to remove {}: {}", file_path, e);
                } else {
                    println!("[DEBUG] Successfully removed: {}", file_path);
                }
            }
        }
    }

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

    println!("Bootstrapping vector registration registry subsystem...");

    let candle_device = Device::Cpu;

    let bert_model_path = Path::new("storage/model.safetensors");
    let bert_config_path = Path::new("storage/config.json");
    let tokenizer_path = Path::new("storage/tokenizer.json");
    let knowledge_txt = Path::new("storage/knowledge.txt");

    // Pass the dynamic device selection instead of the hardcoded CPU flag
    let mut vector_registry = storage::VectorRegistry::bootstrap(
        knowledge_txt,
        bert_model_path,
        bert_config_path,
        tokenizer_path,
        &candle_device,
    )?;

    let storage_dir = Path::new("storage/session.json");
    let mut session = ChatSession::load_or_create(storage_dir)?;

    // --- Step 3: Extract User Input from CLI Arguments ---
    let prompt_arg = if let Some(pos) = args.iter().position(|x| x == "--prompt") {
        if pos + 1 < args.len() {
            Some(args[pos + 1].as_str())
        } else {
            anyhow::bail!("CLI Error: Missing text payload after --prompt flag");
        }
    } else {
        None
    };

    if let Some(prompt_raw) = prompt_arg {
        let prompt_str = prompt_raw.trim();
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
        let execution_prompt = format_prompt(
            preset.template_kind,
            system_instruction,
            context_info.as_ref(),
            &history_context,
        );

        // --- DEBUG PROMPT INJECTION START ---
        println!("\n[DEBUG] === FULL TEXT SENT TO LLAMA.CPP ===");
        println!("{}", execution_prompt);
        println!("[DEBUG] ===================================\n");
        // --- DEBUG PROMPT INJECTION END ---

        print!("Assistant: ");
        std::io::stdout().flush()?;

        let mut ctx = engine.new_context(preset)?;
        if let Err(e) = engine.generate(&mut ctx, &execution_prompt, &mut session, preset, |token| {
            print!("{}", token);
            std::io::stdout().flush()?;
            Ok(())
        }) {
            println!("Generation sequence encountered an error: {:?}", e);
        } else {
            // Persist state updates to disk before application lifecycle termination
            if let Err(e) = session.save(storage_dir) {
                println!("Warning: Failed to flush chat history session to state ledger: {:?}", e);
            }
        }
    } else {
        // Instantiate the persistent context workspace once before starting the chat loop
        let mut ctx = engine.new_context(preset)?;

        // Initialize rustyline editor helper instance
        let mut rl = rustyline::DefaultEditor::new()?;

        loop {
            let readline = rl.readline("User > ");

            let input_raw = match readline {
                Ok(line) => line,
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    println!("Session interrupted via Ctrl-C signal.");
                    break;
                }
                Err(rustyline::error::ReadlineError::Eof) => {
                    println!("Session closed via EOF execution path.");
                    break;
                }
                Err(err) => {
                    anyhow::bail!("REPL Terminal Input Failure: {:?}", err);
                }
            };

            let input_trimmed = input_raw.trim();
            if input_trimmed.is_empty() {
                continue;
            }

            // Save non-empty strings into systemic command memory buffers
            rl.add_history_entry(input_trimmed)?;

            // --- INLINE SYSTEM COMMANDS PROCESSING ---
            if input_trimmed == "/exit" {
                println!("Exiting application cycle.");
                break;
            }

            if input_trimmed == "/clear" {
                session.messages.clear();
                println!("[System] Context history successfully flushed.");
                continue;
            }

            // Perform dynamic RAG context lookup for the current turn
            let context_info = vector_registry.find_relevant_context(input_trimmed)?;

            let facts_payload = match &context_info {
                Some(text) if !text.is_empty() => Some(text),
                _ => None,
            };

            session.add_message("user", input_trimmed);

            let mut raw_history = String::new();
            for msg in &session.messages {
                raw_history.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
            }

            let formatted_prompt = format_prompt(
                preset.template_kind,
                SYSTEM_PROMPT,
                facts_payload,
                &raw_history,
            );

            print!("Assistant: ");
            std::io::stdout().flush()?;

            engine.generate(&mut ctx, &formatted_prompt, &mut session, preset, |token| {
                print!("{}", token);
                std::io::stdout().flush()?;
                Ok(())
            })?;

            if let Err(e) = session.save(storage_dir) {
                println!("Warning: Failed to auto-save session: {:?}", e);
            }
        }
    }

    Ok(())
}

// At the very end of src/main.rs
fn format_prompt(
    template_kind: &str,
    system_instruction: &str,
    facts: Option<&String>,
    history_context: &str,
) -> String {
    // Format optional RAG payload to be appended alongside the immediate execution context
    let rag_marker = if let Some(context_payload) = facts {
        format!("\nContext from knowledge base:\n{}\n", context_payload.trim())
    } else {
        String::new()
    };

    match template_kind {
        "phi3" => {
            // Standard layout for Phi-3 profile architectures
            if !rag_marker.is_empty() {
                format!(
                    "<s><|system|>\n{}<|end|>\n{}{}\n<|assistant|>\n",
                    system_instruction,
                    history_context,
                    rag_marker
                )
            } else {
                format!(
                    "<s><|system|>\n{}<|end|>\n{}\n<|assistant|>\n",
                    system_instruction,
                    history_context
                )
            }
        }
        "chatml" | _ => {
            if !rag_marker.is_empty() {
                format!(
                    "<|im_start|>system\n{}<|im_end|>\n{}{}\n<|im_start|>assistant\n",
                    system_instruction,
                    history_context,
                    rag_marker
                )
            } else {
                format!(
                    "<|im_start|>system\n{}<|im_end|>\n{}\n<|im_start|>assistant\n",
                    system_instruction,
                    history_context
                )
            }
        }
    }
}

