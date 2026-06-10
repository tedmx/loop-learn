#[allow(dead_code)]
#[derive(Clone)]
pub struct PresetInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub vram_limit: &'static str,
    pub repo_id: &'static str,
    pub filename: &'static str,
    pub eos_token: u32,
    pub template_kind: &'static str, 
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f32,
    pub n_ctx: u32, 
    pub max_tokens: u32,
}

pub const PRESETS: &[PresetInfo] = &[
    // --- 6 GB VRAM Profiles (Laptop RTX 3060) ---

    PresetInfo {
        name: "qwen-1.5b-f16",
        description: "Qwen2.5-1.5B GGUF (Unquantized F16 precision, great for lightweight fast tests)",
        vram_limit: "6 GB",
        repo_id: "Qwen/Qwen2.5-1.5B-Instruct",
        filename: "qwen2.5-1.5b-instruct-f16.gguf",
        eos_token: 151645,
        template_kind: "chatml",
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        n_ctx: 2048,
        max_tokens: 512,
    },
    PresetInfo {
        name: "qwen-3b-q4",
        description: "Qwen2.5-3B GGUF (Recommended 4-bit standard profile for 6GB VRAM)",
        vram_limit: "6 GB",
        repo_id: "Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename: "qwen2.5-3b-instruct-q4_k_m.gguf",
        eos_token: 151645,
        template_kind: "chatml",
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        n_ctx: 2048,
        max_tokens: 512,
    },

    // --- 12 GB VRAM Profiles (RTX 2060 Super) ---

    // Temporarily disabled due to strict Meta Hugging Face repository gated access restrictions.
    // Uncomment only if explicit explicit user authorization and token access are cleared.
    /* PresetInfo {
        name: "llama-3.1-8b-q8",
        description: "Llama-3.1-8B GGUF (Meta's industry standard, high-fidelity 8-bit, multilingual)",
        vram_limit: "12 GB",
        repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename: Some("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
        dtype: DType::F32,
        tokenizer_repo: "meta-llama/Llama-3.1-8B-Instruct",
        eos_token: 128009, // Llama 3.1 <|eot_id|> token ID
        template_kind: "llama3",
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
    }, */
    PresetInfo {
        name: "qwen-7b-q3",
        description: "Qwen2.5-7B GGUF (Maximized parameters via efficient 3-bit quantization for 6GB constraints)",
        vram_limit: "12 GB",
        repo_id: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename: "qwen2.5-7b-instruct-q3_k_m.gguf",
        eos_token: 151645, // Standard Qwen2.5 <|im_end|> token ID
        template_kind: "chatml",
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        n_ctx: 4096,
        max_tokens: 1024,
    },
    // Phi-3 preset consumes MORE than 6 GB and produces Out-of-memory errors, unless we use complex technologies like Flash Attention. 
    PresetInfo {
        name: "phi-3-mini-q4",
        description: "Phi-3-Mini-4K GGUF (Microsoft's 3.8B model, high token-rate, clean 4-bit)",
        vram_limit: "12 GB",
        repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
        filename: "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        eos_token: 32000, // Standard Phi-3 <|end|> token ID
        template_kind: "phi3",
        temperature: 0.7,
        top_p: 0.85,
        repetition_penalty: 1.1,
        n_ctx: 4096,
        max_tokens: 1024,
    },
    PresetInfo {
        name: "qwen-14b-q4",
        description: "Qwen2.5-14B GGUF (Heavyweight 4-bit, maximum knowledge capacity)",
        vram_limit: "12 GB",
        repo_id: "bartowski/Qwen2.5-14B-Instruct-GGUF",
        filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        eos_token: 151645,
        template_kind: "chatml",
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        n_ctx: 4096,
        max_tokens: 1024,
    },
];

pub fn list_presets() {
    println!("=== AVAILABLE MODEL PRESETS ===");
    println!("\n[Hardware Target: 6 GB VRAM Limit]");
    for preset in PRESETS.iter().filter(|p| p.vram_limit == "6 GB") {
        println!("  * {:<16} - {}", preset.name, preset.description);
    }
    println!("\n[Hardware Target: 12 GB VRAM Limit]");
    for preset in PRESETS.iter().filter(|p| p.vram_limit == "12 GB") {
        println!("  * {:<16} - {}", preset.name, preset.description);
    }
    println!("===============================");
}
