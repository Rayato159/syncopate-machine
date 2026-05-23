//! Interactive chat with a trained Syncopate NPC-chat model.
//!
//! # Quick start
//!
//! ```sh
//! # First, train a model:
//! cargo run --release --example train_with_tokenizer -- \
//!     --train-dir examples/data --run-dir runs/my-model --steps 5000
//!
//! # Then chat with Mew:
//! cargo run --release --example chat_with_tokenizer -- \
//!     --run-dir runs/my-model --npc mew
//!
//! # One-shot mode:
//! cargo run --release --example chat_with_tokenizer -- \
//!     --run-dir runs/my-model --npc jin --prompt "สวัสดี"
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use sentencepiece_rs::SentencePieceProcessor;
use serde::Deserialize;
use std::fs;
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

    fn decode(&self, ids: &[u32]) -> String {
        let ids: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
        self.proc.decode_ids(&ids).unwrap_or_default()
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

// ---------------------------------------------------------------------------
// Persona prompt contract
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpcId {
    Mew,
    Jin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum ChatLanguage {
    Thai,
    English,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LanguageMode {
    Auto,
    Thai,
    English,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptContract {
    Plain,
    NpcLangV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum RelationState {
    Low,
    Neutral,
    Close,
}

impl NpcId {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "mew" | "หมิว" => Ok(Self::Mew),
            "jin" | "จิน" => Ok(Self::Jin),
            other => bail!("unknown npc '{other}'; use mew or jin"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Mew => "mew",
            Self::Jin => "jin",
        }
    }

    #[allow(dead_code)]
    fn display_name(self, language: ChatLanguage) -> &'static str {
        match (self, language) {
            (Self::Mew, ChatLanguage::Thai) => "หมิว",
            (Self::Mew, ChatLanguage::English) => "Mew",
            (Self::Jin, ChatLanguage::Thai) => "จิน",
            (Self::Jin, ChatLanguage::English) => "Jin",
        }
    }

    #[allow(dead_code)]
    fn traits(self, language: ChatLanguage) -> &'static str {
        match (self, language) {
            (Self::Mew, ChatLanguage::Thai) => {
                "เด็กเรียน สาวแว่น ขี้อาย ลังเล แต่จริงใจมาก และแอบชอบธันวามานาน"
            }
            (Self::Mew, ChatLanguage::English) => {
                "studious, glasses girl, shy, hesitant, very sincere, secretly likes Thanwa for a long time"
            }
            (Self::Jin, ChatLanguage::Thai) => {
                "เด็กกิจกรรม สายลุย กล้าแสดงออก ตรงไปตรงมา ปากร้ายนิดๆ แต่แอบชอบธันวามาก"
            }
            (Self::Jin, ChatLanguage::English) => {
                "active school event girl, bold, direct, brave, sharp-tongued, secretly likes Thanwa a lot"
            }
        }
    }
}

#[allow(dead_code)]
impl ChatLanguage {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "th" | "thai" => Ok(Self::Thai),
            "en" | "eng" | "english" => Ok(Self::English),
            other => bail!("unknown language '{other}'; use th or en"),
        }
    }

    fn token(self) -> &'static str {
        match self {
            Self::Thai => "<th>",
            Self::English => "<en>",
        }
    }
}

impl LanguageMode {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "th" | "thai" => Ok(Self::Thai),
            "en" | "eng" | "english" => Ok(Self::English),
            other => bail!("unknown language mode '{other}'; use auto, th, or en"),
        }
    }

    fn resolve(self, player_message: &str) -> Option<ChatLanguage> {
        match self {
            Self::Auto => detect_text_language(player_message),
            Self::Thai => Some(ChatLanguage::Thai),
            Self::English => Some(ChatLanguage::English),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Thai => "th",
            Self::English => "en",
        }
    }
}

impl PromptContract {
    fn load(run_dir: &Path, value: &str) -> Result<Self> {
        if value.trim().eq_ignore_ascii_case("auto") {
            let path = run_dir.join("prompt_contract.txt");
            if !path.exists() {
                return Ok(Self::Plain);
            }
            let text = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            return Self::parse(text.trim());
        }
        Self::parse(value)
    }

    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "plain" => Ok(Self::Plain),
            "npc-lang-v1" | "npc_lang_v1" | "lang" => Ok(Self::NpcLangV1),
            other => bail!("unknown prompt contract '{other}'; use auto, plain, or npc-lang-v1"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::NpcLangV1 => "npc-lang-v1",
        }
    }
}

#[allow(dead_code)]
impl RelationState {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "neutral" => Ok(Self::Neutral),
            "close" => Ok(Self::Close),
            other => bail!("unknown relation '{other}'; use low, neutral, or close"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Neutral => "neutral",
            Self::Close => "close",
        }
    }
}

#[allow(dead_code)]
fn default_scene(language: ChatLanguage) -> &'static str {
    match language {
        ChatLanguage::Thai => "ห้องเรียนที่เอาโต๊ะกั้นประตู",
        ChatLanguage::English => "classroom barricade",
    }
}

fn build_runtime_prompt(
    npc: NpcId,
    player_message: &str,
    contract: PromptContract,
    language: Option<ChatLanguage>,
) -> String {
    match contract {
        PromptContract::Plain => format!("<{}>\n{}", npc.label(), player_message),
        PromptContract::NpcLangV1 => {
            let language = language.unwrap_or(ChatLanguage::English);
            format!(
                "<{}>\n{}\n{}",
                npc.label(),
                language.token(),
                player_message
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer discovery
// ---------------------------------------------------------------------------

fn find_tokenizer(run_dir: &Path) -> Result<PathBuf> {
    let p = run_dir.join("tokenizer.model");
    if p.exists() {
        return Ok(p);
    }

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
// Sampling
// ---------------------------------------------------------------------------

fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    recent_tokens: &[u32],
    blocked_token_ids: &[u32],
) -> u32 {
    use rand::Rng;
    let mut scores = logits.to_vec();

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

    for &tok in blocked_token_ids {
        let idx = tok as usize;
        if idx < scores.len() {
            scores[idx] = f32::NEG_INFINITY;
        }
    }

    if temperature <= 0.0 {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (idx, &score) in scores.iter().enumerate() {
            if score > best.1 {
                best = (idx, score);
            }
        }
        return best.0 as u32;
    }

    for score in &mut scores {
        *score /= temperature;
    }

    let mut indexed: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .collect();
    if indexed.is_empty() {
        return 0;
    }
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k.max(1).min(indexed.len()));

    let max_val = indexed[0].1;
    let exps: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, score)| (idx, (score - max_val).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, value)| *value).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return indexed[0].0 as u32;
    }

    let mut rng = rand::thread_rng();
    let r: f32 = rng.r#gen();
    let mut cumulative = 0.0f32;
    for &(idx, exp_value) in &exps {
        cumulative += exp_value / sum;
        if r <= cumulative {
            return idx as u32;
        }
    }
    exps.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

fn blocked_token_ids(sp: &SpTokenizer, vocab_size: usize, eos_id: Option<u32>) -> Vec<u32> {
    let mut blocked = Vec::new();
    for id in 0..vocab_size as u32 {
        if Some(id) == eos_id {
            continue;
        }
        let piece = sp.id_to_piece(id);
        if id == 0 || id == 1 || id == 2 || (piece.starts_with("<0x") && piece.ends_with('>')) {
            blocked.push(id);
        }
    }
    blocked
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct NpcReply {
    message: String,
    mood: String,
    relation_point: i32,
}

fn parse_reply(text: &str) -> Option<NpcReply> {
    let trimmed = text.trim();
    if let Ok(reply) = serde_json::from_str::<NpcReply>(trimmed) {
        return Some(reply);
    }

    let object = extract_reply_json(trimmed)?;
    serde_json::from_str::<NpcReply>(object).ok()
}

fn extract_reply_json(text: &str) -> Option<&str> {
    for (start, _) in text.match_indices('{') {
        let Some(object) = extract_json_object_from(text, start) else {
            continue;
        };
        if serde_json::from_str::<NpcReply>(object).is_ok() {
            return Some(object);
        }
    }

    None
}

fn extract_json_object_from(text: &str, start: usize) -> Option<&str> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return Some(&text[start..end]);
                }
            }
            _ => {}
        }
    }

    None
}

fn print_reply(npc: NpcId, generated: &str, show_raw: bool) {
    println!("{}>", npc.label());
    if let Some(reply) = parse_reply(generated) {
        println!("message: {}", reply.message);
        println!("mood: {}", reply.mood);
        println!("point: {}", reply.relation_point);
        if show_raw {
            println!(
                "raw: {}",
                extract_reply_json(generated).unwrap_or_else(|| generated.trim())
            );
        }
    } else {
        println!("raw: {}", generated.trim());
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "chat_with_tokenizer",
    about = "Chat with a trained Syncopate NPC-chat model using persona prompts"
)]
struct Args {
    /// Run directory from train_with_tokenizer (contains checkpoints/).
    #[arg(long, default_value = "runs/my-model")]
    run_dir: PathBuf,

    /// Checkpoint path. Defaults to run_dir/checkpoints/latest.mpk.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// NPC to talk with: mew or jin. If omitted in interactive mode, asks first.
    #[arg(long)]
    npc: Option<String>,

    /// Prompt for one-shot mode. If omitted, starts interactive mode.
    #[arg(long)]
    prompt: Option<String>,

    /// Reply language mode: auto, th, or en.
    #[arg(long, default_value = "auto")]
    lang: String,

    /// Runtime prompt contract: auto, plain, or npc-lang-v1.
    #[arg(long, default_value = "auto")]
    prompt_contract: String,

    /// Max tokens to generate per response.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Sampling temperature (0 = greedy, 1.0 = normal, >1 = more random).
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    /// Top-k sampling: only consider top K most likely tokens.
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Repetition penalty (>1.0 penalizes repeated tokens).
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,

    /// Print raw generated JSON text after parsed fields.
    #[arg(long, default_value_t = false)]
    show_raw: bool,

    /// Retry sampling when parsed message language does not match the input.
    #[arg(long, default_value_t = 4)]
    language_retries: usize,

    /// Temperature used for language-match retries.
    #[arg(long, default_value_t = 0.8)]
    retry_temperature: f32,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let mut language_mode = LanguageMode::parse(&args.lang)?;
    let prompt_contract = PromptContract::load(&args.run_dir, &args.prompt_contract)?;

    let ckpt = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.run_dir.join("checkpoints/latest.mpk"));
    anyhow::ensure!(ckpt.exists(), "checkpoint not found: {}", ckpt.display());

    let tok_path = find_tokenizer(&args.run_dir)?;
    let sp = SpTokenizer::load(&tok_path)?;
    eprintln!("tokenizer: {}", tok_path.display());

    let model = ChatModel::load(&ckpt)?;
    eprintln!(
        "loaded model: {} params",
        model.config().estimated_parameter_count()
    );
    let seq_len = model.config().seq_len;
    let vocab_size = model.config().vocab_size;
    eprintln!("seq_len={seq_len}, vocab_size={vocab_size}");

    let eos_id = sp.eos_id();
    let blocked_token_ids = blocked_token_ids(&sp, vocab_size, eos_id);
    eprintln!(
        "sampling: temperature={:.2}, top_k={}, repetition_penalty={:.2}",
        args.temperature, args.top_k, args.repetition_penalty
    );
    eprintln!(
        "prompt_contract={}, lang={}",
        prompt_contract.label(),
        language_mode.label()
    );
    eprintln!();

    let generate = |prompt_text: &str,
                    max_new_tokens: usize,
                    temperature: f32,
                    top_k: usize,
                    repetition_penalty: f32,
                    force_json_prefix: bool|
     -> Result<String> {
        let prompt_ids = sp.encode(prompt_text);
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt tokenized to empty sequence");
        }

        let mut output_ids = prompt_ids;
        let mut generated_ids: Vec<u32> = Vec::new();
        if force_json_prefix {
            generated_ids = sp.encode("{");
            output_ids.extend_from_slice(&generated_ids);
        }
        let mut full_text = sp.decode(&generated_ids);
        let mut recent_tokens: Vec<u32> = Vec::new();
        const RECENT_WINDOW: usize = 24;

        for _ in 0..max_new_tokens.saturating_sub(generated_ids.len()) {
            let last_index = context_last_index(output_ids.len(), seq_len)?;
            let logits_tensor = model.predict_logits(&output_ids)?;
            let last_logits = logits_tensor
                .slice([0..1, last_index..last_index + 1, 0..vocab_size])
                .reshape([vocab_size]);
            let logits_data = last_logits.into_data();
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap_or_default();

            let next_token = sample_token(
                &logits_vec,
                temperature,
                top_k,
                repetition_penalty,
                &recent_tokens,
                &blocked_token_ids,
            );

            if Some(next_token) == eos_id {
                break;
            }

            output_ids.push(next_token);
            generated_ids.push(next_token);
            recent_tokens.push(next_token);
            if recent_tokens.len() > RECENT_WINDOW {
                recent_tokens.remove(0);
            }

            full_text = sp.decode(&generated_ids);

            if parse_reply(&full_text).is_some() {
                break;
            }
        }

        Ok(full_text)
    };

    let generate_for_player =
        |npc: NpcId, player_message: &str, language_mode: LanguageMode| -> Result<String> {
            let target_language = language_mode.resolve(player_message);
            let prompt =
                build_runtime_prompt(npc, player_message, prompt_contract, target_language);
            let mut generated = generate(
                &format!("{prompt}\n"),
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.repetition_penalty,
                false,
            )?;
            if language_ok(&generated, target_language) {
                return Ok(generated);
            }

            let greedy = generate(
                &format!("{prompt}\n"),
                args.max_new_tokens,
                0.0,
                1,
                args.repetition_penalty,
                false,
            )?;
            if language_ok(&greedy, target_language) {
                return Ok(greedy);
            }
            if parse_reply(&generated).is_none() && parse_reply(&greedy).is_some() {
                generated = greedy;
            }

            let forced = generate(
                &format!("{prompt}\n"),
                args.max_new_tokens,
                0.0,
                1,
                args.repetition_penalty,
                true,
            )?;
            if language_ok(&forced, target_language) {
                return Ok(forced);
            }
            if parse_reply(&generated).is_none() && parse_reply(&forced).is_some() {
                generated = forced;
            }

            let retry_temperature = args.retry_temperature.max(args.temperature);
            let retry_top_k = args.top_k.max(20);
            for _ in 0..args.language_retries {
                let retry = generate(
                    &format!("{prompt}\n"),
                    args.max_new_tokens,
                    retry_temperature,
                    retry_top_k,
                    args.repetition_penalty,
                    true,
                )?;
                if language_ok(&retry, target_language) {
                    return Ok(retry);
                }
                if parse_reply(&generated).is_none() && parse_reply(&retry).is_some() {
                    generated = retry;
                }
            }

            Ok(generated)
        };

    let mut npc = match (&args.npc, &args.prompt) {
        (Some(value), _) => NpcId::parse(value)?,
        (None, Some(_)) => NpcId::Mew,
        (None, None) => ask_npc()?,
    };

    if let Some(player_message) = &args.prompt {
        println!("me>\n{player_message}\n");
        let generated = generate_for_player(npc, player_message, language_mode)?;
        print_reply(npc, &generated, args.show_raw);
        return Ok(());
    }

    eprintln!(
        "talking to {}. commands: /npc mew|jin, /lang auto|th|en, /quit",
        npc.label()
    );

    loop {
        print!("me> ");
        io::stdout().flush()?;

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
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
            continue;
        }
        if input == "quit" || input == "exit" || input == "q" || input == "/quit" {
            eprintln!("bye!");
            break;
        }
        if handle_command(input, &mut npc, &mut language_mode)? {
            continue;
        }

        let generated = generate_for_player(npc, input, language_mode)?;
        println!();
        print_reply(npc, &generated, args.show_raw);
    }

    Ok(())
}

fn ask_npc() -> Result<NpcId> {
    loop {
        print!("npc (mew/jin)> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        match NpcId::parse(input.trim()) {
            Ok(npc) => return Ok(npc),
            Err(err) => eprintln!("{err}"),
        }
    }
}

fn handle_command(input: &str, npc: &mut NpcId, language_mode: &mut LanguageMode) -> Result<bool> {
    let Some(rest) = input.strip_prefix('/') else {
        return Ok(false);
    };

    let mut parts = rest.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or("").trim();
    let value = parts.next().unwrap_or("").trim();

    match command {
        "npc" => {
            *npc = NpcId::parse(value)?;
            println!("npc set to {}", npc.label());
        }
        "lang" => {
            *language_mode = LanguageMode::parse(value)?;
            println!("language set to {}", language_mode.label());
        }
        "help" => {
            println!("commands: /npc mew|jin, /lang auto|th|en, /quit");
        }
        other => bail!("unknown command '/{other}'"),
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn context_last_index(token_count: usize, seq_len: usize) -> Result<usize> {
    if token_count == 0 {
        bail!("prompt tokenized to empty sequence");
    }
    Ok(token_count.min(seq_len) - 1)
}

fn detect_text_language(text: &str) -> Option<ChatLanguage> {
    if text.chars().any(is_thai_char) {
        return Some(ChatLanguage::Thai);
    }
    if text.chars().any(|ch| ch.is_ascii_alphabetic()) {
        return Some(ChatLanguage::English);
    }
    None
}

fn language_ok(generated: &str, target: Option<ChatLanguage>) -> bool {
    let Some(target) = target else {
        return true;
    };
    let Some(reply) = parse_reply(generated) else {
        return false;
    };
    match target {
        ChatLanguage::Thai => reply.message.chars().any(is_thai_char),
        ChatLanguage::English => {
            reply.message.chars().any(|ch| ch.is_ascii_alphabetic())
                && !reply.message.chars().any(is_thai_char)
        }
    }
}

fn is_thai_char(ch: char) -> bool {
    ('\u{0e00}'..='\u{0e7f}').contains(&ch)
}
