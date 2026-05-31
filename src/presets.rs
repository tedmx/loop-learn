use candle_core::DType;

#[derive(Clone, Copy)]
pub enum ModelKind {
    Standard,
    Gguf,
}

#[derive(Clone)]
pub struct PresetInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub vram_limit: &'static str,
    pub kind: ModelKind,
    pub repo_id: &'static str,
    pub filename: Option<&'static str>,
    pub dtype: DType,
    pub tokenizer_repo: &'static str,
    pub eos_token: u32,
    pub template_kind: &'static str, 
}

pub const PRESETS: &[PresetInfo] = &[
    // --- 6 GB VRAM Profiles (Laptop RTX 3060) ---
    PresetInfo {
        name: "qwen-1.5b-bf16",
        description: "Qwen2.5-1.5B Standard (Optimized precision BF16 execution)",
        vram_limit: "6 GB",
        kind: ModelKind::Standard,
        repo_id: "Qwen/Qwen2.5-1.5B-Instruct",
        filename: None,
        dtype: DType::BF16,
        tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct",
        eos_token: 151645,
        template_kind: "chatml",
    },
    PresetInfo {
        name: "qwen-3b-q4",
        description: "Qwen2.5-3B GGUF (Recommended 4-bit standard profile)",
        vram_limit: "6 GB",
        kind: ModelKind::Gguf,
        repo_id: "Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename: Some("qwen2.5-3b-instruct-q4_k_m.gguf"),
        dtype: DType::F32,
        tokenizer_repo: "Qwen/Qwen2.5-3B-Instruct",
        eos_token: 151645,
        template_kind: "chatml",
    },
    PresetInfo {
        name: "qwen-3b-q8",
        description: "Qwen2.5-3B GGUF (High-density 8-bit quantization, pristine logic & multilingual)",
        vram_limit: "6 GB",
        kind: ModelKind::Gguf,
        repo_id: "bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename: Some("Qwen2.5-3B-Instruct-Q8_0.gguf"),
        dtype: DType::F32,
        tokenizer_repo: "Qwen/Qwen2.5-3B-Instruct",
        eos_token: 151645,
        template_kind: "chatml",
    },
    PresetInfo {
        name: "phi-3-mini-q4",
        description: "Phi-3-Mini-4K GGUF (Microsoft's 3.8B model, high token-rate, clean 4-bit)",
        vram_limit: "6 GB",
        kind: ModelKind::Gguf,
        repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
        filename: Some("Phi-3-mini-4k-instruct-Q4_K_M.gguf"),
        dtype: DType::F32,
        tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct",
        eos_token: 32000, // Standard Phi-3 <|end|> token ID
        template_kind: "phi3",
    },
    // --- 12 GB VRAM Profiles (RTX 2060 Super) ---
    PresetInfo {
        name: "llama-3.1-8b-q8",
        description: "Llama-3.1-8B GGUF (Meta's industry standard, high-fidelity 8-bit, multilingual)",
        vram_limit: "12 GB",
        kind: ModelKind::Gguf,
        repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename: Some("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
        dtype: DType::F32,
        tokenizer_repo: "meta-llama/Llama-3.1-8B-Instruct",
        eos_token: 128009, // Llama 3.1 <|eot_id|> token ID
        template_kind: "llama3",
    },
    PresetInfo {
        name: "qwen-14b-q4",
        description: "Qwen2.5-14B GGUF (Heavyweight 4-bit, maximum knowledge capacity)",
        vram_limit: "12 GB",
        kind: ModelKind::Gguf,
        repo_id: "bartowski/Qwen2.5-14B-Instruct-GGUF",
        filename: Some("Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
        dtype: DType::F32,
        tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct",
        eos_token: 151645,
        template_kind: "chatml",
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
