//! Interactive chat with a trained Syncopate NPC-chat model.
//!
//! # Quick start
//!
//! ```sh
//! # First, train a model:
//! cargo run --release --example train_with_tokenizer -- \
//!     --train-dir examples/data --run-dir runs/my-model --steps 5000
//!
//! # Then chat with it:
//! cargo run --release --example chat_with_tokenizer -- --run-dir runs/my-model
//!
//! # One-shot mode:
//! cargo run --release --example chat_with_tokenizer -- \
//!     --run-dir runs/my-model --prompt "สวัสดี"
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use sentencepiece_rs::SentencePieceProcessor;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use syncopate_machine::prelude::*;

// ---------------------------------------------------------------------------
// SentencePiece adapter
// ---------------------------------------------------------------------------

struct SpTokenizer {
    proc: SentencePieceProcessor,
}

impl SpTokenizer {
    fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            proc: SentencePieceProcessor::open(path)
                .with_context(|| format!("failed to load {}", path.display()))?,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.proc
            .encode_to_ids(text)
            .unwrap_or_default()
            .into_iter()
            .map(|id| id as u32)
            .collect()
    }

    fn eos_id(&self) -> Option<u32> {
        self.proc.eos_id().map(|id| id as u32)
    }

    fn id_to_piece(&self, id: u32) -> String {
        self.proc
            .id_to_piece(id as usize)
            .map(|s| s.to_owned())
            .unwrap_or_default()
    }
}

/// Decode a token piece from SentencePiece.
///
/// Handles three cases:
/// - `<0xNN>` → byte fallback token → convert to actual char
/// - `▁` prefix → SentencePiece space marker → replace with space
/// - anything else → return as-is
fn piece_to_text(piece: &str) -> String {
    // SentencePiece byte fallback: <0xNN> → raw byte
    if piece.starts_with("<0x")
        && piece.ends_with('>')
        && piece.len() == 6
        && let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16)
    {
        return (byte_val as char).to_string();
    }

    // SentencePiece uses U+2581 (▁) as space marker
    if let Some(rest) = piece.strip_prefix('\u{2581}') {
        return format!(" {rest}");
    }

    piece.to_owned()
}

// ---------------------------------------------------------------------------
// Tokenizer discovery
// ---------------------------------------------------------------------------

fn find_tokenizer(run_dir: &Path) -> Result<PathBuf> {
    // 1. Inside run dir (copied by train_with_tokenizer)
    let p = run_dir.join("tokenizer.model");
    if p.exists() {
        return Ok(p);
    }
    // 2. Bundled in examples/data/
    let p = PathBuf::from("examples/data/tokenizer.model");
    if p.exists() {
        return Ok(p);
    }
    anyhow::bail!(
        "tokenizer.model not found in {} or examples/data/",
        run_dir.display()
    )
}

// ---------------------------------------------------------------------------
// Sampling with temperature + top-k + repetition penalty
// ---------------------------------------------------------------------------

/// Sample from logits with temperature, top-k filtering, and repetition penalty.
/// Returns a sampled token ID.
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    recent_tokens: &[u32],
) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    if temperature <= 0.0 {
        // Greedy — but still apply repetition penalty
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &v) in logits.iter().enumerate() {
            let mut score = v;
            if recent_tokens.contains(&(i as u32)) && repetition_penalty > 1.0 {
                score /= repetition_penalty;
            }
            if score > best.1 {
                best = (i, score);
            }
        }
        return best.0 as u32;
    }

    // Apply repetition penalty
    let mut scores: Vec<f32> = logits.to_vec();
    for &tok in recent_tokens {
        let idx = tok as usize;
        if idx < scores.len() && repetition_penalty > 1.0 {
            if scores[idx] > 0.0 {
                scores[idx] /= repetition_penalty;
            } else {
                scores[idx] *= repetition_penalty;
            }
        }
    }

    // Temperature scaling
    let mut scores: Vec<f32> = scores.iter().map(|s| s / temperature).collect();

    // Suppress special tokens: <unk>=0, <0x00>=1, </s>=2
    scores[0] = f32::NEG_INFINITY;
    if scores.len() > 1 {
        scores[1] = f32::NEG_INFINITY;
    }
    if scores.len() > 2 {
        scores[2] = f32::NEG_INFINITY; // </s> — don't generate unless forced
    }

    // Suppress byte fallback tokens (<0xNN>) — they produce garbage output.
    // SentencePiece with byte_fallback places them at IDs 3..258 (after <unk>, <0x00>, </s>).
    // These tokens represent raw bytes, not meaningful text — the model should use
    // proper subword tokens instead. Suppressing forces the model to pick real tokens.
    for i in 3..scores.len().min(259) {
        scores[i] = f32::NEG_INFINITY;
    }

    // Top-k filtering: keep only the top-k highest scores
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Softmax over remaining
    let max_val = indexed[0].1;
    let exps: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, s)| (idx, (s - max_val).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, e)| *e).sum();
    let probs: Vec<(usize, f32)> = exps.iter().map(|&(idx, e)| (idx, e / sum)).collect();

    // Weighted random sample
    let r: f32 = rng.r#gen();
    let mut cumulative = 0.0f32;
    for &(idx, p) in &probs {
        cumulative += p;
        if r <= cumulative {
            return idx as u32;
        }
    }
    probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "chat_with_tokenizer",
    about = "Chat with a trained Syncopate NPC-chat model using streaming output"
)]
struct Args {
    /// Run directory from train_with_tokenizer (contains checkpoints/).
    #[arg(long, default_value = "runs/my-model")]
    run_dir: PathBuf,

    /// Checkpoint path. Defaults to run_dir/checkpoints/latest.mpk.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// One-shot prompt. If omitted, starts interactive mode.
    #[arg(long)]
    prompt: Option<String>,

    /// Max tokens to generate per response.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Sampling temperature (0 = greedy, 1.0 = normal, >1 = more random).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-k sampling: only consider top K most likely tokens.
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Repetition penalty (>1.0 penalizes repeated tokens).
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Resolve checkpoint path ---
    let ckpt = args
        .checkpoint
        .unwrap_or_else(|| args.run_dir.join("checkpoints/latest.mpk"));
    anyhow::ensure!(ckpt.exists(), "checkpoint not found: {}", ckpt.display());

    // --- Load tokenizer ---
    let tok_path = find_tokenizer(&args.run_dir)?;
    let sp = SpTokenizer::load(&tok_path)?;
    eprintln!("tokenizer: {}", tok_path.display());

    // --- Load model ---
    let model = ChatModel::load(&ckpt)?;
    eprintln!(
        "loaded model: {} params",
        model.config().estimated_parameter_count()
    );
    let seq_len = model.config().seq_len;
    let vocab_size = model.config().vocab_size;
    eprintln!("seq_len={seq_len}, vocab_size={vocab_size}");

    let eos_id = sp.eos_id();
    let pad_token_id: u32 = 0;

    eprintln!(
        "sampling: temperature={:.2}, top_k={}, repetition_penalty={:.2}",
        args.temperature, args.top_k, args.repetition_penalty
    );
    eprintln!();

    // --- Generate function (manual autoregressive with sampling) ---
    let generate = |prompt_text: &str,
                    max_new_tokens: usize,
                    temperature: f32,
                    top_k: usize,
                    repetition_penalty: f32|
     -> Result<String> {
        let prompt_ids = sp.encode(prompt_text);
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt tokenized to empty sequence");
        }

        let mut output_ids = prompt_ids.clone();
        let mut full_text = String::new();
        let mut recent_tokens: Vec<u32> = Vec::new();
        const RECENT_WINDOW: usize = 16; // track last N tokens for repetition penalty

        let mut consecutive_eos = 0usize;
        const MAX_CONSECUTIVE_EOS: usize = 3;

        for _i in 0..max_new_tokens {
            // Run one forward pass to get logits
            let input = context_window(&output_ids, seq_len, pad_token_id);
            let logits_tensor = model.predict_logits(&input)?;
            // logits_tensor shape: [1, seq_len, vocab_size]
            // Extract the last position's logits: slice [0, seq_len-1, :]
            let last_logits = logits_tensor
                .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
                .reshape([vocab_size]);
            let logits_data = last_logits.into_data();
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap_or_default();

            // Sample next token
            let next_token = sample_token(
                &logits_vec,
                temperature,
                top_k,
                repetition_penalty,
                &recent_tokens,
            );

            // Stop at EOS
            if Some(next_token) == eos_id {
                consecutive_eos += 1;
                if full_text.is_empty() && consecutive_eos < MAX_CONSECUTIVE_EOS {
                    continue; // skip EOS at the very start (up to N times)
                }
                break;
            }
            consecutive_eos = 0;

            output_ids.push(next_token);
            recent_tokens.push(next_token);
            if recent_tokens.len() > RECENT_WINDOW {
                recent_tokens.remove(0);
            }

            // Decode and print
            let piece = sp.id_to_piece(next_token);
            let text = piece_to_text(&piece);
            full_text.push_str(&text);
            print!("{text}");
            io::stdout().flush().ok();
        }

        println!(); // newline after generation
        Ok(full_text)
    };

    // --- One-shot mode ---
    if let Some(prompt) = &args.prompt {
        // Send prompt as-is (no role prefixes)
        let _ = generate(
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )?;
        return Ok(());
    }

    // --- Interactive mode ---
    eprintln!("interactive mode — type a message, press Enter. Type 'quit' or 'exit' to stop.");
    let stdin = io::stdin();
    loop {
        print!("me> ");
        io::stdout().flush()?;

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => {
                eprintln!("\n[EOF] exiting.");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\n[stdin error: {e}] exiting.");
                break;
            }
        }
        let input = input.trim();
        if input.is_empty() {
            continue; // skip blank lines instead of exiting
        }
        if input == "quit" || input == "exit" || input == "q" {
            eprintln!("bye!");
            break;
        }

        // Send prompt directly as raw text, matching the training format:
        //   training sequence: "คำถาม\nคำตอบ</s>"
        // The model sees the question + newline as context, then generates the answer.
        // We append "\n" so the model knows the question is done and should start answering.
        let prompt = format!("{input}\n");
        print!("ai> ");
        io::stdout().flush()?;
        let _ = generate(
            &prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pad/truncate context to seq_len (left-pad with pad_token_id).
fn context_window(tokens: &[u32], seq_len: usize, pad_token_id: u32) -> Vec<u32> {
    if tokens.len() >= seq_len {
        tokens[(tokens.len() - seq_len)..].to_vec()
    } else {
        let pad_count = seq_len - tokens.len();
        let mut padded = vec![pad_token_id; pad_count];
        padded.extend_from_slice(tokens);
        padded
    }
}
