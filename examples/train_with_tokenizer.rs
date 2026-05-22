//! Train a Syncopate NPC-chat model with SentencePiece tokenization and produce a
//! full training report.
//!
//! # Quick start
//!
//! ```sh
//! # Train 10M params, 10k steps
//! cargo run --release --example train_with_tokenizer -- \
//!     --train-dir examples/data --run-dir runs/10m-10k --budget 10m --steps 10000
//!
//! # Smaller model for quick testing
//! cargo run --release --example train_with_tokenizer -- \
//!     --train-dir examples/data --run-dir runs/test --budget 1m --steps 500
//!
//! # Then chat with the result:
//! cargo run --release --example chat_with_tokenizer -- \
//!     --run-dir runs/10m-10k
//!
//! # Generate a loss plot from the CSV (requires Python + matplotlib):
//! python examples/plot_loss.py runs/10m-10k/loss.csv
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use rayon::prelude::*;
use sentencepiece_rs::SentencePieceProcessor;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
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

    fn decode(&self, ids: &[u32]) -> String {
        let ids: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
        self.proc.decode_ids(&ids).unwrap_or_default()
    }

    fn vocab_size(&self) -> usize {
        self.proc.model().vocab_size()
    }

    fn eos_id(&self) -> Option<u32> {
        self.proc.eos_id().map(|id| id as u32)
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "train_with_tokenizer",
    about = "Train a Syncopate NPC-chat LM with SentencePiece and produce a full report"
)]
struct Args {
    /// Directory with tokenizer.model + .txt/.jsonl training files.
    #[arg(long, default_value = "examples/data")]
    train_dir: PathBuf,

    /// Output directory for checkpoints, reports, loss CSV.
    #[arg(long, default_value = "runs/my-model")]
    run_dir: PathBuf,

    /// Parameter budget: 1m, 5m, 10m, 50m, 100m.
    #[arg(long, default_value = "10m")]
    budget: String,

    /// Total optimizer steps.
    #[arg(long, default_value_t = 10_000)]
    steps: usize,

    /// Batch size.
    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    /// Sequence length.
    #[arg(long, default_value_t = 128)]
    seq_len: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,

    /// Minimum learning rate after cosine decay.
    #[arg(long, default_value_t = 3e-5)]
    min_lr: f64,

    /// Linear warmup steps before cosine decay.
    #[arg(long, default_value_t = 200)]
    warmup_steps: usize,

    /// AdamW weight decay.
    #[arg(long, default_value_t = 0.1)]
    weight_decay: f64,

    /// Gradient clipping norm (0 disables clipping).
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f64,

    /// Attention kernel: softmax (default) or higher-order.
    #[arg(long, default_value = "softmax")]
    attention_kernel: String,

    /// Fraction of data used for validation (0.0–1.0).
    #[arg(long, default_value_t = 0.1)]
    val_split: f64,

    /// Number of prompt tokens to generate for inference latency test.
    #[arg(long, default_value_t = 20)]
    latency_tokens: usize,

    /// Print loss every N steps.
    #[arg(long, default_value_t = 100)]
    log_interval: usize,

    /// Evaluate val loss every N steps and save best_val.mpk.
    #[arg(long, default_value_t = 500)]
    eval_interval: usize,

    /// Save a checkpoint every N steps (0 = only save best).
    #[arg(long, default_value_t = 0)]
    checkpoint_interval: usize,

    /// Maximum number of samples to load (0 = all). Useful for huge text files.
    #[arg(long, default_value_t = 0)]
    max_samples: usize,

    /// Skip training — load existing checkpoint and only run evaluation + report.
    #[arg(long, default_value_t = false)]
    eval_only: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split text into sentences, preserving the terminating punctuation.
/// Handles '.', '!', '?' as sentence boundaries. Also handles '"' after
/// punctuation (e.g. `said, \"hello.\"`).
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut start = 0;
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];
        if c == '.' || c == '!' || c == '?' {
            // Include the punctuation
            let mut end = i + 1;
            // Also consume a closing quote after punctuation: ." !" ?"
            if end < n && (chars[end] == '"' || chars[end] == '\'' || chars[end] == '\u{201d}') {
                end += 1;
            }
            let sentence: String = chars[start..end].iter().collect();
            let trimmed = sentence.trim().to_owned();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            start = end;
            i = end;
        } else {
            i += 1;
        }
    }

    // Handle remaining text (no terminating punctuation)
    if start < n {
        let remaining: String = chars[start..].iter().collect();
        let trimmed = remaining.trim().to_owned();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
    }

    sentences
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

fn load_samples(dir: &Path, max_samples: usize) -> Result<Vec<(String, String)>> {
    let mut samples = Vec::new();

    let is_maxed = |len: usize| max_samples > 0 && len >= max_samples;

    for entry in fs::read_dir(dir).with_context(|| format!("cannot read {}", dir.display()))? {
        if is_maxed(samples.len()) {
            break;
        }
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "csv" => {
                // CSV with `prompt` and `response` columns.
                // Simple Q&A format: prompt = question, response = answer.
                // No role prefixes — raw text in, raw text out.
                let text = fs::read_to_string(&path)
                    .with_context(|| format!("cannot read {}", path.display()))?;
                let mut reader = csv::ReaderBuilder::new()
                    .flexible(true)
                    .from_reader(text.as_bytes());

                // Validate headers
                let headers = reader.headers()?.clone();
                let has_prompt = headers.iter().any(|h| h.trim() == "prompt");
                let has_response = headers.iter().any(|h| h.trim() == "response");
                if !has_prompt || !has_response {
                    // Skip CSVs that don't have the expected columns
                    // (might be some other CSV we don't care about)
                    continue;
                }

                let prompt_idx = headers.iter().position(|h| h.trim() == "prompt").unwrap();
                let response_idx = headers.iter().position(|h| h.trim() == "response").unwrap();

                for result in reader.records() {
                    if is_maxed(samples.len()) {
                        break;
                    }
                    let record = match result {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let prompt = record.get(prompt_idx).unwrap_or("").trim().to_owned();
                    let response = record.get(response_idx).unwrap_or("").trim().to_owned();
                    if prompt.is_empty() || response.is_empty() {
                        continue;
                    }
                    samples.push((prompt, response));
                }
            }
            "txt" => {
                // Plain text corpus: blank lines separate samples.
                // Uses streaming BufReader so huge files (e.g. 1.8GB TinyStories)
                // don't need to fit in memory.
                //
                // Each sample spans multiple lines (joined with space).
                // Stories are split into input (first ~60%) and output (last ~40%)
                // so the model learns story continuation with loss masking.
                let file = fs::File::open(&path)
                    .with_context(|| format!("cannot open {}", path.display()))?;
                let reader = std::io::BufReader::new(file);
                let mut current_lines: Vec<String> = Vec::new();

                let flush = |lines: &mut Vec<String>, out: &mut Vec<(String, String)>| {
                    if lines.is_empty() {
                        return;
                    }
                    let story = lines.join(" ");
                    lines.clear();
                    let story = story.trim().to_owned();
                    if story.is_empty() {
                        return;
                    }

                    // Split story into sentences preserving punctuation.
                    // We find sentence boundaries by locating '.', '!', '?' chars.
                    let sentences = split_sentences(&story);

                    if sentences.len() < 2 {
                        // Too short to split — use whole thing as response only
                        out.push((String::new(), story));
                        return;
                    }

                    let split_point = (sentences.len() as f64 * 0.6).ceil() as usize;
                    let split_point = split_point.max(1).min(sentences.len() - 1);
                    let input = sentences[..split_point].join(" ");
                    let output = sentences[split_point..].join(" ");

                    if output.is_empty() {
                        out.push((String::new(), story));
                    } else {
                        out.push((input, output));
                    }
                };

                let mut line_count: u64 = 0;
                for line_result in reader.lines() {
                    if is_maxed(samples.len()) {
                        break;
                    }
                    let line = match line_result {
                        Ok(l) => l,
                        Err(_) => continue,
                    };
                    line_count += 1;
                    if line_count.is_multiple_of(1_000_000) {
                        eprintln!(
                            "  streaming {}: {line_count} lines, {} stories so far",
                            path.file_name().unwrap_or_default().to_string_lossy(),
                            samples.len()
                        );
                    }
                    if line.trim().is_empty() {
                        flush(&mut current_lines, &mut samples);
                    } else {
                        current_lines.push(line.trim().to_owned());
                    }
                }
                flush(&mut current_lines, &mut samples); // last sample
                eprintln!(
                    "  loaded {} stories from {}",
                    samples.len(),
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
            "jsonl" => {
                let text = fs::read_to_string(&path)
                    .with_context(|| format!("cannot read {}", path.display()))?;
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                        // Format 1: {"text": "..."}
                        // No masking — full text is the response.
                        if let Some(s) = val.get("text").and_then(|v| v.as_str()) {
                            samples.push((String::new(), s.to_owned()));
                        }
                        // Format 2: {"input": "...", "output": "..."}
                        // Also accepts prompt/response aliases used by exported chat corpora.
                        else if let Some(response) = val
                            .get("output")
                            .or_else(|| val.get("response"))
                            .and_then(|v| v.as_str())
                        {
                            let prompt = val
                                .get("input")
                                .or_else(|| val.get("prompt"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .trim()
                                .to_owned();
                            let response = response.trim().to_owned();
                            if !response.is_empty() {
                                samples.push((prompt, response));
                            }
                        }
                        // Format 3: {"messages": [{"role": "...", "content": "..."}, ...]}
                        // Extract raw content without role prefixes.
                        else if let Some(messages) =
                            val.get("messages").and_then(|v| v.as_array())
                        {
                            let mut prompt_parts: Vec<String> = Vec::new();
                            let mut response_parts: Vec<String> = Vec::new();
                            for msg in messages {
                                let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
                                let content =
                                    msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                                if content.is_empty() {
                                    continue;
                                }
                                match role {
                                    "assistant" => response_parts.push(content.to_owned()),
                                    _ => prompt_parts.push(content.to_owned()),
                                }
                            }
                            if !response_parts.is_empty() {
                                let prompt = prompt_parts.join("\n");
                                let response = response_parts.join("\n");
                                samples.push((prompt, response));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(samples)
}

fn parse_budget(s: &str) -> Result<SyncopateParameterBudget> {
    match s.to_lowercase().as_str() {
        "1m" => Ok(SyncopateParameterBudget::Params1M),
        "5m" => Ok(SyncopateParameterBudget::Params5M),
        "10m" => Ok(SyncopateParameterBudget::Params10M),
        "50m" => Ok(SyncopateParameterBudget::Params50M),
        "100m" => Ok(SyncopateParameterBudget::Params100M),
        other => bail!("unknown budget '{other}'; use 1m, 5m, 10m, 50m, or 100m"),
    }
}

fn parse_attention_kernel(s: &str) -> Result<AttentionKernel> {
    match s.to_lowercase().as_str() {
        "higher-order" | "higher_order" | "ho" => Ok(AttentionKernel::HigherOrder),
        "softmax" | "default" => Ok(AttentionKernel::Softmax),
        other => bail!("unknown attention kernel '{other}'; use 'softmax' or 'higher-order'"),
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct RunMeta {
    step: usize,
    loss: f64,
    params: usize,
    model_config: SyncopateModelConfig,
}

#[derive(Serialize, Deserialize, Clone)]
struct TrainReport {
    /// Model configuration.
    budget: String,
    parameter_count: usize,
    seq_len: usize,
    batch_size: usize,
    learning_rate: f64,
    total_steps: usize,
    /// Wall-clock training time in seconds.
    train_duration_secs: f64,
    /// Steps per second.
    steps_per_sec: f64,
    /// Final training loss.
    final_train_loss: f64,
    /// Best (lowest) training loss observed.
    best_train_loss: f64,
    /// Validation results.
    val: Option<EvalMetrics>,
    /// Test results.
    test: Option<EvalMetrics>,
    /// Inference latency.
    inference: Option<InferenceMetrics>,
    /// Data statistics.
    train_samples: usize,
    val_samples: usize,
    test_samples: usize,
    total_tokens: usize,
    /// Device used.
    device: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct EvalMetrics {
    loss: f64,
    perplexity: f64,
    accuracy: f64,
    tokens: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct InferenceMetrics {
    /// Average time per generated token in milliseconds.
    avg_ms_per_token: f64,
    /// Total tokens generated during the benchmark.
    tokens_generated: usize,
    /// Total wall-clock time in seconds.
    total_secs: f64,
}

// ---------------------------------------------------------------------------
// Tokenization cache
// ---------------------------------------------------------------------------

/// Cache file format stored as JSON.
#[derive(Serialize, Deserialize)]
struct TokenCache {
    /// Hash of (train_dir file names/sizes + max_samples + tokenizer bytes).
    cache_key: String,
    /// Tokenized (prompt, response) pairs.
    pairs: Vec<(Vec<u32>, Vec<u32>)>,
}

/// Compute a cache key from the data source configuration.
fn compute_cache_key(train_dir: &Path, max_samples: usize, tokenizer_path: &Path) -> String {
    let mut hasher = DefaultHasher::new();

    // Hash file names and sizes from train_dir
    if let Ok(entries) = fs::read_dir(train_dir) {
        let mut file_infos: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let size = e.metadata().ok().map(|m| m.len()).unwrap_or(0);
                (name, size)
            })
            .collect();
        file_infos.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, size) in &file_infos {
            name.hash(&mut hasher);
            size.hash(&mut hasher);
        }
    }
    max_samples.hash(&mut hasher);

    // Hash tokenizer file contents
    if let Ok(bytes) = fs::read(tokenizer_path) {
        bytes.hash(&mut hasher);
    }

    format!("{:016x}", hasher.finish())
}

/// Try to load tokenized pairs from cache. Returns `None` on any mismatch or error.
fn try_load_token_cache(path: &Path, expected_key: &str) -> Option<Vec<(Vec<u32>, Vec<u32>)>> {
    let data = fs::read_to_string(path).ok()?;
    let cache: TokenCache = serde_json::from_str(&data).ok()?;
    if cache.cache_key != expected_key {
        return None;
    }
    Some(cache.pairs)
}

/// Save tokenized pairs to cache.
fn save_token_cache(path: &Path, cache_key: &str, pairs: &[(Vec<u32>, Vec<u32>)]) -> Result<()> {
    let cache = TokenCache {
        cache_key: cache_key.to_owned(),
        pairs: pairs.to_vec(),
    };
    let json = serde_json::to_string(&cache)?;
    fs::write(path, json)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Validate train dir ---
    let tokenizer_path = args.train_dir.join("tokenizer.model");
    if !tokenizer_path.exists() {
        bail!("tokenizer.model not found in {}", args.train_dir.display());
    }

    // --- Prepare output dirs ---
    let ckpt_dir = args.run_dir.join("checkpoints");
    fs::create_dir_all(&ckpt_dir)
        .with_context(|| format!("cannot create {}", ckpt_dir.display()))?;

    // --- Load tokenizer ---
    let sp = SpTokenizer::load(&tokenizer_path)?;
    let vocab_size = sp.vocab_size();
    let eos_id = sp.eos_id();
    println!("tokenizer: vocab_size={vocab_size} eos={eos_id:?}");

    // --- Load & tokenize data (with caching) ---
    let samples = load_samples(&args.train_dir, args.max_samples)?;
    if samples.is_empty() {
        bail!("no training samples found in {}", args.train_dir.display());
    }
    println!("loaded {} samples", samples.len());

    // Detect chat data: any sample with a non-empty prompt.
    let has_chat_data = samples.iter().any(|(prompt, _)| !prompt.is_empty());
    if has_chat_data {
        println!("detected chat-format data — using loss masking for prompt tokens");
    }

    // Compute a cache key from: train_dir files + max_samples + tokenizer hash.
    // If the cache file exists and the key matches, skip tokenization.
    let cache_path = args.run_dir.join("token_cache.json");
    let cache_key = compute_cache_key(&args.train_dir, args.max_samples, &tokenizer_path);

    let mut chat_pairs: Vec<(Vec<u32>, Vec<u32>)> =
        if let Some(pairs) = try_load_token_cache(&cache_path, &cache_key) {
            println!(
                "loaded {} tokenized pairs from cache ({})",
                pairs.len(),
                cache_path.display()
            );
            pairs
        } else {
            // Tokenize (prompt, response) pairs separately — in parallel with rayon.
            //
            // The training sequence will be: prompt_tokens + "\n" + response_tokens + EOS
            // The \n separator helps the model distinguish where the question ends
            // and the answer begins. Loss masking ensures only response tokens are learned.
            eprintln!("tokenizing {} samples (parallel)...", samples.len());
            let tokenize_start = Instant::now();
            let newline_ids = sp.encode("\n");
            let progress = AtomicUsize::new(0);
            let total = samples.len();

            let pairs: Vec<(Vec<u32>, Vec<u32>)> = samples
                .par_iter()
                .enumerate()
                .filter_map(|(_, (prompt, response))| {
                    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if done.is_multiple_of(1000) {
                        eprintln!(
                            "  tokenized {}/{} ({:.1}s)",
                            done,
                            total,
                            tokenize_start.elapsed().as_secs_f64()
                        );
                    }
                    let prompt_ids = if prompt.is_empty() {
                        Vec::new()
                    } else {
                        let mut ids = sp.encode(prompt);
                        ids.extend_from_slice(&newline_ids);
                        ids
                    };
                    let mut response_ids = sp.encode(response);
                    if let Some(eos) = eos_id {
                        response_ids.push(eos);
                    }
                    if prompt_ids.len() + response_ids.len() >= 2 {
                        Some((prompt_ids, response_ids))
                    } else {
                        None
                    }
                })
                .collect();

            eprintln!(
                "tokenized {} pairs in {:.1}s",
                pairs.len(),
                tokenize_start.elapsed().as_secs_f64()
            );

            // Save cache for future runs.
            eprintln!("saving token cache...");
            if let Err(e) = save_token_cache(&cache_path, &cache_key, &pairs) {
                eprintln!("warning: failed to save token cache: {e}");
            } else {
                println!("token cache saved: {}", cache_path.display());
            }
            eprintln!("cache save done.");

            pairs
        };

    if chat_pairs.is_empty() {
        bail!("all samples tokenized to <2 tokens — cannot train");
    }

    // Deduplicate
    eprintln!("deduplicating {} pairs...", chat_pairs.len());
    let dedup_start = Instant::now();
    chat_pairs.sort();
    chat_pairs.dedup();
    eprintln!(
        "dedup done: {} pairs in {:.1}s",
        chat_pairs.len(),
        dedup_start.elapsed().as_secs_f64()
    );

    // Shuffle after dedup so train/val/test split is random and
    // sequential batching sees diverse data each step.
    eprintln!("shuffling {} pairs...", chat_pairs.len());
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    chat_pairs.shuffle(&mut thread_rng());
    eprintln!("shuffled.");

    // Build flat sequences for evaluation (prompt + response concatenated).
    let sequences: Vec<Vec<u32>> = chat_pairs
        .iter()
        .map(|(p, r)| {
            let mut seq = p.clone();
            seq.extend_from_slice(r);
            seq
        })
        .collect();

    let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
    println!(
        "{} token sequences ({} tokens total, deduped)",
        sequences.len(),
        total_tokens
    );

    // --- Split data: 80% train, 10% val, 10% test ---
    let n = chat_pairs.len();
    let val_count = ((n as f64 * args.val_split).ceil() as usize)
        .max(1)
        .min(n / 2);
    let test_count = val_count.min(n - val_count);
    let train_count = n.saturating_sub(val_count + test_count);

    let (train_pairs, rest) = chat_pairs.split_at(train_count);
    let (val_pairs, test_pairs) = rest.split_at(rest.len().min(val_count));

    // Flat sequences for evaluation.
    let val_seqs: Vec<Vec<u32>> = val_pairs
        .iter()
        .map(|(p, r)| {
            let mut seq = p.clone();
            seq.extend_from_slice(r);
            seq
        })
        .collect();
    let test_seqs: Vec<Vec<u32>> = test_pairs
        .iter()
        .map(|(p, r)| {
            let mut seq = p.clone();
            seq.extend_from_slice(r);
            seq
        })
        .collect();

    println!(
        "split: {} train, {} val, {} test sequences",
        train_pairs.len(),
        val_seqs.len(),
        test_seqs.len()
    );

    // --- Build model config ---
    eprintln!("building model config...");
    let budget = parse_budget(&args.budget)?;
    let attention_kernel = parse_attention_kernel(&args.attention_kernel)?;
    let config = SyncopateModelConfig::for_parameter_budget(budget, vocab_size, args.seq_len)
        .with_attention_kernel(attention_kernel);
    let param_count = config.estimated_parameter_count();
    println!(
        "model: {} params, budget={}, attention={:?}",
        param_count, args.budget, attention_kernel
    );

    // --- Save config.json in checkpoint dir ---
    let config_json = serde_json::to_string_pretty(&config)?;
    fs::write(ckpt_dir.join("config.json"), &config_json)?;

    // --- Copy tokenizer to run dir ---
    fs::copy(&tokenizer_path, args.run_dir.join("tokenizer.model"))
        .with_context(|| "failed to copy tokenizer to run dir")?;

    // --- Device ---
    eprintln!("initializing device...");
    let device_start = Instant::now();
    let device = auto_device()?;
    let device_name = device_label(&device);
    println!(
        "device: {device_name} (took {:.1}s)",
        device_start.elapsed().as_secs_f64()
    );

    let final_ckpt = ckpt_dir.join("latest.mpk");

    // --- Loss CSV path (used by both train mode and final message) ---
    let loss_csv_path = args.run_dir.join("loss.csv");

    // --- Train or eval-only ---
    let (train_steps, final_loss, train_params, train_secs, steps_per_sec, best_loss) =
        if args.eval_only {
            // === EVAL-ONLY MODE: skip training, load existing checkpoint ===
            anyhow::ensure!(
                final_ckpt.exists(),
                "--eval-only requires an existing checkpoint at {}",
                final_ckpt.display()
            );
            let meta_path = ckpt_dir.join("latest.json");
            let meta: RunMeta = if meta_path.exists() {
                let json = fs::read_to_string(&meta_path)
                    .with_context(|| format!("cannot read {}", meta_path.display()))?;
                serde_json::from_str(&json)
                    .with_context(|| format!("cannot parse {}", meta_path.display()))?
            } else {
                anyhow::bail!(
                    "--eval-only requires {} from a previous training run",
                    meta_path.display()
                );
            };
            println!("skipping training, loading checkpoint from previous run");
            println!(
                "  step={}  loss={:.6}  params={}",
                meta.step, meta.loss, meta.params
            );

            (
                meta.step,
                meta.loss as f32,
                meta.params,
                0.0,
                0.0,
                meta.loss,
            )
        } else {
            // === FULL TRAINING MODE ===
            let mut loss_csv = fs::File::create(&loss_csv_path).with_context(|| {
                format!("cannot create loss CSV at {}", loss_csv_path.display())
            })?;
            writeln!(loss_csv, "step,loss")?;

            eprintln!("building trainer (model init on GPU)...");
            let trainer_start = Instant::now();
            let mut trainer = Trainer::builder()
                .vocab_size(vocab_size)
                .budget(budget)
                .device({
                    #[cfg(feature = "cuda")]
                    {
                        device.clone()
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        device
                    }
                })
                .batch_size(args.batch_size)
                .seq_len(args.seq_len)
                .steps(args.steps)
                .learning_rate(args.lr)
                .min_learning_rate(args.min_lr)
                .warmup_steps(args.warmup_steps)
                .weight_decay(args.weight_decay)
                .grad_clip_norm((args.grad_clip > 0.0).then_some(args.grad_clip))
                .attention_kernel(attention_kernel)
                .checkpoint_dir(ckpt_dir.to_string_lossy().into_owned())
                .checkpoint_interval(args.checkpoint_interval)
                .build()?;
            eprintln!(
                "trainer built in {:.1}s",
                trainer_start.elapsed().as_secs_f64()
            );

            let log_interval = args.log_interval;
            let eval_interval = args.eval_interval;
            let mut bl = f64::MAX;
            let mut best_vl = f64::MAX;
            let _loss_values: Vec<(usize, f64)> = Vec::new();

            println!(
                "\ntraining {} steps (eval_interval={})...",
                args.steps, eval_interval
            );
            let train_start = Instant::now();

            let rpt = if has_chat_data {
                // Flatten val_pairs into flat sequences for evaluation.
                trainer.train_on_chat_sequences_with_validation(
                train_pairs,
                &val_seqs,
                eval_interval,
                |step, loss, val_loss| {
                    let loss_f64 = loss as f64;
                    if loss_f64 < bl {
                        bl = loss_f64;
                    }

                    // Write train loss to CSV.
                    let _ = writeln!(&mut loss_csv, "{step},{loss_f64}");
                    let _ = loss_csv.flush();

                    // Log val loss when available.
                    if let Some(vl) = val_loss {
                        let vl_f64 = vl as f64;
                        let improved = if vl_f64 < best_vl {
                            best_vl = vl_f64;
                            " *best*"
                        } else {
                            ""
                        };
                        println!(
                            "  step {}/{}  val_loss={:.6}{}  train_loss={:.6}  best_train={:.6}",
                            step + 1, args.steps, vl_f64, improved, loss_f64, bl,
                        );
                    }

                    // Log progress at log_interval.
                    if step == 0 || ((step + 1) % log_interval == 0 && val_loss.is_none()) {
                        let elapsed = train_start.elapsed().as_secs_f64();
                        let sps = if step > 0 {
                            (step + 1) as f64 / elapsed
                        } else {
                            0.0
                        };
                        println!(
                            "  step {}/{}  loss={:.6}  best={:.6}  {:.1} steps/s",
                            step + 1,
                            args.steps,
                            loss_f64,
                            bl,
                            sps
                        );
                    }
                },
            )
            } else {
                // No chat data — fall back to plain token-sequence training.
                let train_seqs: Vec<Vec<u32>> = train_pairs
                    .iter()
                    .map(|(p, r)| {
                        let mut seq = p.clone();
                        seq.extend_from_slice(r);
                        seq
                    })
                    .collect();
                trainer.train_on_token_sequences_with_callback(&train_seqs, |step, loss| {
                    let loss_f64 = loss as f64;
                    if loss_f64 < bl {
                        bl = loss_f64;
                    }

                    // Write to CSV and flush immediately so data survives a crash.
                    let _ = writeln!(&mut loss_csv, "{step},{loss_f64}");
                    let _ = loss_csv.flush();

                    // Log progress
                    if step == 0 || (step + 1) % log_interval == 0 {
                        let elapsed = train_start.elapsed().as_secs_f64();
                        let sps = if step > 0 {
                            (step + 1) as f64 / elapsed
                        } else {
                            0.0
                        };
                        println!(
                            "  step {}/{}  loss={:.6}  best={:.6}  {:.1} steps/s",
                            step + 1,
                            args.steps,
                            loss_f64,
                            bl,
                            sps
                        );
                    }
                })
            }?;

            let train_duration = train_start.elapsed();
            let ts = train_duration.as_secs_f64();
            let sps = args.steps as f64 / ts;

            // Flush CSV
            drop(loss_csv);

            println!("\ntraining complete in {:.1}s ({:.1} steps/s)", ts, sps);
            println!(
                "  final loss: {:.6}  best loss: {:.6} (step {})  params: {}",
                rpt.final_loss,
                rpt.best_loss,
                rpt.best_loss_step + 1,
                rpt.parameter_count
            );

            // Save the final weights as latest.mpk.
            trainer.save_checkpoint(final_ckpt.to_str().unwrap())?;
            println!("checkpoint: {}", final_ckpt.display());

            let meta = RunMeta {
                step: rpt.steps,
                loss: rpt.final_loss as f64,
                params: rpt.parameter_count,
                model_config: config.clone(),
            };
            fs::write(
                ckpt_dir.join("latest.json"),
                serde_json::to_string_pretty(&meta)?,
            )?;

            // Free trainer + model from GPU before evaluation.
            drop(trainer);

            (
                rpt.steps,
                rpt.final_loss,
                rpt.parameter_count,
                ts,
                sps,
                rpt.best_loss as f64,
            )
        };

    // --- Evaluate on val and test sets ---
    println!("\nevaluating...");

    // Build an inference-only model for evaluation.
    // Load with Autodiff backend first (for parameter loading), then strip
    // the autodiff wrapper via .valid() so forward passes don't track gradients.
    // This prevents VRAM from growing on every eval batch.
    use burn::module::AutodiffModule;
    let eval_model = {
        let mut m = DefaultSyncopateModel::new(config.clone(), &device)?;
        m.load_parameters(&final_ckpt)?;
        m.valid() // SyncopateModel<Autodiff<Cuda>> -> SyncopateModel<Cuda>
    };
    let inner_device = device;

    let val_metrics = if !val_seqs.is_empty() {
        println!("  validation set ({} sequences)...", val_seqs.len());
        let result =
            eval_model.evaluate_on_sequences(&val_seqs, args.seq_len, 4, 0, &inner_device)?;
        println!(
            "    loss={:.4}  ppl={:.2}  accuracy={:.2}%  ({} tokens)",
            result.loss,
            result.perplexity,
            result.accuracy * 100.0,
            result.total_tokens
        );
        Some(EvalMetrics {
            loss: result.loss as f64,
            perplexity: result.perplexity as f64,
            accuracy: result.accuracy,
            tokens: result.total_tokens,
        })
    } else {
        None
    };

    let test_metrics = if !test_seqs.is_empty() {
        println!("  test set ({} sequences)...", test_seqs.len());
        let result =
            eval_model.evaluate_on_sequences(&test_seqs, args.seq_len, 4, 0, &inner_device)?;
        println!(
            "    loss={:.4}  ppl={:.2}  accuracy={:.2}%  ({} tokens)",
            result.loss,
            result.perplexity,
            result.accuracy * 100.0,
            result.total_tokens
        );
        Some(EvalMetrics {
            loss: result.loss as f64,
            perplexity: result.perplexity as f64,
            accuracy: result.accuracy,
            tokens: result.total_tokens,
        })
    } else {
        None
    };

    // Free GPU memory before loading ChatModel (avoids CUDA OOM)
    drop(eval_model);

    // --- Measure inference latency ---
    println!("\nmeasuring inference latency...");
    // Use a prompt that matches the training data format.
    // For TinyStories (story continuation): start of a story.
    // For chat data (Q&A): a question followed by newline.
    let prompt = if has_chat_data {
        // Use the first training sample's prompt if available
        match train_pairs.first() {
            Some((p, _)) if !p.is_empty() => {
                // Decode prompt tokens back to text for display
                sp.decode(p)
            }
            _ => "Once upon a time, there was a little".to_owned(),
        }
    } else {
        "Once upon a time, there was a little".to_owned()
    };
    println!("  sample prompt: {prompt}");
    let prompt_ids = sp.encode(&prompt);
    let chat_model = ChatModel::load(&final_ckpt)?;

    let latency_start = Instant::now();
    let output = chat_model.generate(
        &prompt_ids,
        GenerationConfig {
            max_new_tokens: args.latency_tokens,
            ..Default::default()
        },
    )?;
    let latency_secs = latency_start.elapsed().as_secs_f64();
    let new_tokens = output.len().saturating_sub(prompt_ids.len());
    let avg_ms_per_token = if new_tokens > 0 {
        latency_secs * 1000.0 / new_tokens as f64
    } else {
        0.0
    };

    let inference_metrics = InferenceMetrics {
        avg_ms_per_token,
        tokens_generated: new_tokens,
        total_secs: latency_secs,
    };

    println!(
        "  {} tokens in {:.3}s = {:.2} ms/token",
        new_tokens, latency_secs, avg_ms_per_token
    );

    // --- Generate sample output ---
    // Decode only the NEW tokens (skip the prompt part)
    let new_token_ids: Vec<u32> = if output.len() > prompt_ids.len() {
        output[prompt_ids.len()..].to_vec()
    } else {
        output.clone()
    };
    let generated_text = sp.decode(&new_token_ids);
    let full_text = sp.decode(&output);
    println!("\nsample output:");
    println!("  prompt:    {prompt}");
    println!("  generated: {generated_text}");
    println!("  full:      {full_text}");

    // --- Write report ---
    let full_report = TrainReport {
        budget: args.budget.clone(),
        parameter_count: train_params,
        seq_len: args.seq_len,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        total_steps: train_steps,
        train_duration_secs: train_secs,
        steps_per_sec,
        final_train_loss: final_loss as f64,
        best_train_loss: best_loss,
        val: val_metrics,
        test: test_metrics,
        inference: Some(inference_metrics),
        train_samples: train_pairs.len(),
        val_samples: val_seqs.len(),
        test_samples: test_seqs.len(),
        total_tokens,
        device: device_name,
    };

    let report_json = serde_json::to_string_pretty(&full_report)?;
    let report_path = args.run_dir.join("report.json");
    fs::write(&report_path, &report_json)?;
    println!("\nreport: {}", report_path.display());

    // --- Write human-readable report markdown ---
    let md = format_report_md(&full_report);
    let report_md_path = args.run_dir.join("report.md");
    fs::write(&report_md_path, &md)?;
    println!("report: {}", report_md_path.display());

    // --- Print loss plot command ---
    println!("\nloss CSV: {}", loss_csv_path.display());
    println!(
        "to generate a loss plot: python examples/plot_loss.py {}",
        loss_csv_path.display()
    );

    println!(
        "\nnext step: cargo run --release --example chat_with_tokenizer -- --run-dir {}",
        args.run_dir.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

fn format_report_md(r: &TrainReport) -> String {
    let mut md = String::new();

    md.push_str("# Training Report\n\n");

    md.push_str("## Configuration\n\n");
    md.push_str("| Parameter | Value |\n|---|---|\n");
    md.push_str(&format!("| Budget | {} |\n", r.budget));
    md.push_str(&format!(
        "| Parameters | {} (~{:.1}M) |\n",
        r.parameter_count,
        r.parameter_count as f64 / 1e6
    ));
    md.push_str(&format!("| Seq Length | {} |\n", r.seq_len));
    md.push_str(&format!("| Batch Size | {} |\n", r.batch_size));
    md.push_str(&format!("| Learning Rate | {} |\n", r.learning_rate));
    md.push_str(&format!("| Total Steps | {} |\n", r.total_steps));
    md.push_str(&format!("| Device | {} |\n", r.device));
    md.push('\n');

    md.push_str("## Data\n\n");
    md.push_str("| Split | Sequences |\n|---|---|\n");
    md.push_str(&format!("| Train | {} |\n", r.train_samples));
    md.push_str(&format!("| Val | {} |\n", r.val_samples));
    md.push_str(&format!("| Test | {} |\n", r.test_samples));
    md.push_str(&format!("| Total Tokens | {} |\n", r.total_tokens));
    md.push('\n');

    md.push_str("## Training\n\n");
    md.push_str("| Metric | Value |\n|---|---|\n");
    md.push_str(&format!("| Duration | {:.1}s |\n", r.train_duration_secs));
    md.push_str(&format!(
        "| Throughput | {:.1} steps/s |\n",
        r.steps_per_sec
    ));
    md.push_str(&format!("| Final Loss | {:.6} |\n", r.final_train_loss));
    md.push_str(&format!("| Best Loss | {:.6} |\n", r.best_train_loss));
    md.push('\n');

    if let Some(val) = &r.val {
        md.push_str("## Validation\n\n");
        md.push_str("| Metric | Value |\n|---|---|\n");
        md.push_str(&format!("| Loss | {:.4} |\n", val.loss));
        md.push_str(&format!("| Perplexity | {:.2} |\n", val.perplexity));
        md.push_str(&format!("| Accuracy | {:.2}% |\n", val.accuracy * 100.0));
        md.push_str(&format!("| Tokens | {} |\n", val.tokens));
        md.push('\n');
    }

    if let Some(test) = &r.test {
        md.push_str("## Test\n\n");
        md.push_str("| Metric | Value |\n|---|---|\n");
        md.push_str(&format!("| Loss | {:.4} |\n", test.loss));
        md.push_str(&format!("| Perplexity | {:.2} |\n", test.perplexity));
        md.push_str(&format!("| Accuracy | {:.2}% |\n", test.accuracy * 100.0));
        md.push_str(&format!("| Tokens | {} |\n", test.tokens));
        md.push('\n');
    }

    if let Some(inf) = &r.inference {
        md.push_str("## Inference\n\n");
        md.push_str("| Metric | Value |\n|---|---|\n");
        md.push_str(&format!(
            "| Avg Latency | {:.2} ms/token |\n",
            inf.avg_ms_per_token
        ));
        md.push_str(&format!(
            "| Tokens Generated | {} |\n",
            inf.tokens_generated
        ));
        md.push_str(&format!("| Total Time | {:.3}s |\n", inf.total_secs));
        md.push('\n');
    }

    md.push_str("## Loss Plot\n\n");
    md.push_str("Generate with: `python examples/plot_loss.py runs/<name>/loss.csv`\n");

    md
}
