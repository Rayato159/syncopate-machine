//! Train a tiny action model on integer ID sequences.
//!
//! Data comes from `examples/action-data/train.txt` and `val.txt` where each
//! line is a space-separated list of integer action IDs (vocab = 0..22).
//!
//! # Quick start
//!
//! ```sh
//! cargo run --release --features cuda --example train_action_model
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use std::fs;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;
use syncopate_machine::prelude::*;

const ACTION_VOCAB_SIZE: usize = 15;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "train_action_model",
    about = "Train a tiny action model on integer ID sequences"
)]
struct Args {
    /// Number of training steps.
    #[arg(long, default_value_t = 2000)]
    steps: usize,

    /// Batch size.
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 0.003)]
    lr: f64,

    /// Sequence length.
    #[arg(long, default_value_t = 16)]
    seq_len: usize,

    /// Directory for checkpoints and config output.
    #[arg(long, default_value = "runs/action-model")]
    checkpoint_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Read a text file where each line contains space-separated u32 action IDs.
fn load_int_sequences(path: &PathBuf) -> Result<Vec<Vec<u32>>> {
    let file = fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut sequences = Vec::new();
    for line in reader.lines() {
        let line = line.context("error reading line")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ids: Vec<u32> = trimmed
            .split_whitespace()
            .map(|s| s.parse::<u32>().context(format!("invalid action id: {s}")))
            .collect::<Result<Vec<u32>>>()?;
        if !ids.is_empty() {
            sequences.push(ids);
        }
    }
    Ok(sequences)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ActionTrainReport {
    model_config: SyncopateModelConfig,
    parameter_count: usize,
    steps: usize,
    batch_size: usize,
    learning_rate: f64,
    seq_len: usize,
    final_train_loss: f32,
    best_train_loss: f32,
    train_duration_secs: f64,
    steps_per_sec: f64,
    train_samples: usize,
    val_samples: usize,
    device: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Load data ---
    let train_path = PathBuf::from("examples/action-data/train.txt");
    let val_path = PathBuf::from("examples/action-data/val.txt");

    println!("loading training data from {}...", train_path.display());
    let train_seqs =
        load_int_sequences(&train_path).with_context(|| "failed to load training data")?;
    if train_seqs.is_empty() {
        bail!("no training sequences found in {}", train_path.display());
    }
    println!("  {} training sequences", train_seqs.len());

    println!("loading validation data from {}...", val_path.display());
    let val_seqs =
        load_int_sequences(&val_path).with_context(|| "failed to load validation data")?;
    println!("  {} validation sequences", val_seqs.len());

    // --- Build model config ---
    let config = SyncopateModelConfig::preset_action(ACTION_VOCAB_SIZE, args.seq_len);
    println!("kernel: Softmax");
    let param_count = config.estimated_parameter_count();
    println!(
        "model: {param_count} params, vocab={}, seq_len={}",
        config.vocab_size, config.seq_len
    );

    // --- Prepare output directory ---
    fs::create_dir_all(&args.checkpoint_dir)
        .with_context(|| format!("cannot create {}", args.checkpoint_dir.display()))?;

    // Save model-config.json for browser loading.
    let config_json = serde_json::to_string_pretty(&config)?;
    let config_path = args.checkpoint_dir.join("model-config.json");
    fs::write(&config_path, &config_json)
        .with_context(|| format!("cannot write {}", config_path.display()))?;
    println!("config: {}", config_path.display());

    // --- Device ---
    let device = auto_device()?;
    let device_name = device_label(&device);
    if !device_name.to_ascii_lowercase().contains("cuda") {
        bail!(
            "train_action_model is CUDA-only. Run with: cargo run --release --features cuda --example train_action_model -- <args>"
        );
    }
    println!("device: {device_name}");

    // --- Create model ---
    let mut model = DefaultSyncopateModel::new(config.clone(), &device)?;

    // --- Training config ---
    let training = ModelTrainingConfig {
        steps: args.steps,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        min_learning_rate: args.lr * 0.1,
        warmup_steps: (args.steps / 10).max(10),
        weight_decay: 0.01,
        grad_clip_norm: Some(1.0),
        pad_action_id: 0,
        checkpoint_dir: Some(args.checkpoint_dir.to_string_lossy().to_string()),
        checkpoint_interval: 0,
    };

    // --- Loss CSV ---
    let loss_csv_path = args.checkpoint_dir.join("loss.csv");
    let mut loss_csv = fs::File::create(&loss_csv_path)
        .with_context(|| format!("cannot create {}", loss_csv_path.display()))?;
    writeln!(loss_csv, "step,loss")?;

    // --- Train ---
    let train_start = Instant::now();
    println!(
        "\ntraining {} steps (batch_size={}, lr={})...",
        args.steps, args.batch_size, args.lr
    );

    let log_interval = (args.steps / 20).max(1);
    let _eval_interval = (args.steps / 5).max(1);

    let report = model.train_action_sequences(&train_seqs, &training, &device, |step, loss| {
        // Write to CSV
        let _ = writeln!(&mut loss_csv, "{step},{loss}");
        let _ = loss_csv.flush();

        if step == 0 || (step + 1) % log_interval == 0 {
            let elapsed = train_start.elapsed().as_secs_f64();
            let sps = if step > 0 {
                (step + 1) as f64 / elapsed
            } else {
                0.0
            };
            println!(
                "  step {}/{}  loss={:.6}  ({:.1} steps/s)",
                step + 1,
                args.steps,
                loss,
                sps
            );
        }
    })?;

    let train_duration = train_start.elapsed();
    let train_secs = train_duration.as_secs_f64();
    let steps_per_sec = args.steps as f64 / train_secs;

    println!(
        "\ntraining complete in {:.1}s ({:.1} steps/s)",
        train_secs, steps_per_sec
    );
    println!(
        "  final loss: {:.6}  best loss: {:.6} (step {})  params: {}",
        report.final_loss,
        report.best_loss,
        report.best_loss_step + 1,
        report.parameter_count
    );

    // --- Save final checkpoint ---
    let ckpt_path = args.checkpoint_dir.join("final.mpk");
    model.save_parameters(&ckpt_path)?;
    println!("checkpoint: {}", ckpt_path.display());

    // --- Evaluate on validation set ---
    if !val_seqs.is_empty() {
        println!("\nevaluating on {} validation sequences...", val_seqs.len());
        use burn::module::AutodiffModule;
        let eval_model = model.valid();
        let result = eval_model.evaluate_on_sequences(
            &val_seqs,
            args.seq_len,
            args.batch_size,
            0,
            &device,
        )?;
        println!(
            "  val loss={:.4}  ppl={:.2}  accuracy={:.2}%  ({} action ids)",
            result.loss,
            result.perplexity,
            result.accuracy * 100.0,
            result.total_predictions
        );
    }

    // --- Save report ---
    let full_report = ActionTrainReport {
        model_config: config,
        parameter_count: report.parameter_count,
        steps: report.steps,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        seq_len: args.seq_len,
        final_train_loss: report.final_loss,
        best_train_loss: report.best_loss,
        train_duration_secs: train_secs,
        steps_per_sec,
        train_samples: train_seqs.len(),
        val_samples: val_seqs.len(),
        device: device_name,
    };

    let report_json = serde_json::to_string_pretty(&full_report)?;
    let report_path = args.checkpoint_dir.join("report.json");
    fs::write(&report_path, &report_json)?;
    println!("\nreport: {}", report_path.display());

    Ok(())
}
