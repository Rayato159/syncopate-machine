use crate::config::MultiscreenConfig;
use crate::error::{Error, Result};
use crate::layout::{ScreenLayout, causal_trim_relevance};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

// =====================================================================
// Training types (from train.rs)
// =====================================================================

/// User-provided training input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrainInput {
    TokenSequences(Vec<Vec<u32>>),
}

impl TrainInput {
    pub fn from_token_sequences(token_sequences: Vec<Vec<u32>>) -> Self {
        Self::TokenSequences(token_sequences)
    }

    #[deprecated(note = "use from_token_sequences for paper-aligned naming")]
    pub fn from_token_ids(token_sequences: Vec<Vec<u32>>) -> Self {
        Self::from_token_sequences(token_sequences)
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::TokenSequences(token_sequences) => token_sequences.is_empty(),
        }
    }
}

/// Summary returned after training.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainReport {
    #[serde(alias = "sequence_count")]
    pub training_sequence_count: usize,
    #[serde(alias = "token_count")]
    pub training_token_count: usize,
    #[serde(alias = "unique_tokens")]
    pub observed_vocab_size: usize,
    #[serde(alias = "screen_count")]
    pub screen_layout_count: usize,
    #[serde(alias = "tile_count")]
    pub screening_tile_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct ScreeningState {
    #[serde(alias = "token_counts", alias = "training_observed_token_counts")]
    pub observed_token_counts: BTreeMap<u32, usize>,
    #[serde(alias = "transitions")]
    next_token_counts: HashMap<u32, BTreeMap<u32, usize>>,
    #[serde(alias = "sequence_count")]
    pub training_sequence_count: usize,
    #[serde(alias = "token_count")]
    pub training_token_count: usize,
    #[serde(alias = "screen_count")]
    pub screen_layout_count: usize,
    #[serde(alias = "tile_count")]
    pub screening_tile_count: usize,
}

impl ScreeningState {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn observe_token_sequence(&mut self, tokens: &[u32], layout: &ScreenLayout) {
        self.training_sequence_count += 1;
        self.training_token_count += tokens.len();
        self.screen_layout_count += layout.screens().len();
        self.screening_tile_count += layout.screening_tile_count();

        for token in tokens {
            *self.observed_token_counts.entry(*token).or_insert(0) += 1;
        }

        for pair in tokens.windows(2) {
            let from = pair[0];
            let to = pair[1];
            *self
                .next_token_counts
                .entry(from)
                .or_default()
                .entry(to)
                .or_insert(0) += 1;
        }
    }

    pub fn report(&self) -> TrainReport {
        TrainReport {
            training_sequence_count: self.training_sequence_count,
            training_token_count: self.training_token_count,
            observed_vocab_size: self.observed_token_counts.len(),
            screen_layout_count: self.screen_layout_count,
            screening_tile_count: self.screening_tile_count,
        }
    }

    pub fn predict_next_token(&self, token: u32) -> Option<u32> {
        self.next_token_counts.get(&token).and_then(|candidates| {
            candidates
                .iter()
                .max_by(|left, right| {
                    let count_order = left.1.cmp(right.1);
                    if count_order == std::cmp::Ordering::Equal {
                        right.0.cmp(left.0)
                    } else {
                        count_order
                    }
                })
                .map(|(token, _count)| *token)
        })
    }

    pub fn fallback_token(&self) -> Option<u32> {
        self.observed_token_counts
            .iter()
            .max_by(|left, right| {
                let count_order = left.1.cmp(right.1);
                if count_order == std::cmp::Ordering::Equal {
                    right.0.cmp(left.0)
                } else {
                    count_order
                }
            })
            .map(|(token, _count)| *token)
    }
}

// =====================================================================
// Inference types (from inference.rs)
// =====================================================================

/// Output from token inference.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceOutput {
    pub output_token_ids: Vec<u32>,
    pub layout: ScreenLayout,
    pub mean_distance_relevance_alpha_d: f32,
}

/// Serialized weights file containing the config and learned state.
///
/// This is the on-disk format. The config is embedded so that
/// `load_weights` can verify it matches the engine's active config.
#[derive(Serialize, Deserialize)]
struct ScreeningWeightsFile {
    config: MultiscreenConfig,
    state: ScreeningState,
    report: TrainReport,
}

/// Main user-facing engine.
#[derive(Clone, Debug)]
pub struct MultiscreenEngine {
    config: MultiscreenConfig,
    state: ScreeningState,
}

impl MultiscreenEngine {
    /// Creates a new engine with the given configuration.
    pub fn new(config: MultiscreenConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            state: ScreeningState::default(),
        })
    }

    /// Creates a new engine by loading a weights file.
    ///
    /// The config embedded in the weights file becomes the engine's config.
    /// This is the easiest way to resume from a previously saved model.
    pub fn from_weights_file(path: impl AsRef<Path>) -> Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))?;
        let file: ScreeningWeightsFile =
            serde_json::from_str(&contents).map_err(|e| Error::Serialization(e.to_string()))?;
        file.config.validate()?;
        Ok(Self {
            config: file.config,
            state: file.state,
        })
    }

    /// Returns the active configuration.
    pub fn config(&self) -> &MultiscreenConfig {
        &self.config
    }

    /// Trains the lightweight transition state from token IDs.
    pub fn train(&mut self, input: TrainInput) -> Result<TrainReport> {
        let sequences = self.resolve_train_input(input)?;
        self.state.clear();

        for sequence in &sequences {
            let layout = ScreenLayout::build(&self.config, sequence.len())?;
            self.state.observe_token_sequence(sequence, &layout);
        }

        Ok(self.state.report())
    }

    /// Runs deterministic token inference.
    pub fn infer_tokens(&self, input_ids: &[u32]) -> Result<InferenceOutput> {
        let layout = ScreenLayout::build(&self.config, input_ids.len())?;
        let limit = self
            .config
            .inference
            .max_inference_tokens
            .unwrap_or(input_ids.len())
            .min(input_ids.len());

        let fallback = self.state.fallback_token();
        let output_token_ids = input_ids
            .iter()
            .take(limit)
            .map(|token| {
                self.state
                    .predict_next_token(*token)
                    .or(fallback)
                    .or_else(|| {
                        self.config
                            .inference
                            .use_input_token_fallback
                            .then_some(*token)
                    })
                    .ok_or_else(|| {
                        Error::Inference(
                            "no trained transition, fallback token, or input fallback available"
                                .into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(InferenceOutput {
            output_token_ids,
            mean_distance_relevance_alpha_d: layout_relevance(&layout, &self.config),
            layout,
        })
    }

    // ------------------------------------------------------------------
    // Weight persistence
    // ------------------------------------------------------------------

    /// Saves the engine's config, learned state, and training report to a JSON
    /// weights file.
    ///
    /// The file embeds the full `MultiscreenConfig` so that `load_weights` can
    /// verify a match before restoring state.
    pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = ScreeningWeightsFile {
            config: self.config.clone(),
            state: self.state.clone(),
            report: self.state.report(),
        };
        let json =
            serde_json::to_string_pretty(&file).map_err(|e| Error::Serialization(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| Error::Io(e.to_string()))?;
        Ok(())
    }

    /// Loads weights from a JSON file into this engine.
    ///
    /// **Config validation:** the config stored in the weights file is compared
    /// against this engine's active config. If they do not match exactly, the
    /// load is **rejected** and a [`Error::WeightsConfigMismatch`] is returned.
    /// This prevents subtle bugs caused by running a state trained under one
    /// configuration through an engine configured differently.
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<TrainReport> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))?;
        let file: ScreeningWeightsFile =
            serde_json::from_str(&contents).map_err(|e| Error::Serialization(e.to_string()))?;

        if file.config != self.config {
            return Err(Error::WeightsConfigMismatch(
                "the config embedded in the weights file does not match the engine's active \
                 config — create a new engine with the correct config first, or use \
                 MultiscreenEngine::from_weights_file() to auto-detect the config"
                    .into(),
            ));
        }

        self.state = file.state;
        Ok(self.state.report())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn resolve_train_input(&self, input: TrainInput) -> Result<Vec<Vec<u32>>> {
        match input {
            TrainInput::TokenSequences(sequences) => Ok(sequences),
        }
    }
}

fn layout_relevance(layout: &ScreenLayout, config: &MultiscreenConfig) -> f32 {
    let mut alpha_d_sum = 0.0;
    let mut relevance_pair_count = 0usize;

    for screen in layout.screens() {
        for tile in &screen.tiles {
            if tile.span.is_empty() {
                continue;
            }
            let query_index_i = tile.span.end - 1;
            for key_index_j in tile.span.start..tile.span.end {
                let distance = query_index_i.abs_diff(key_index_j) as f32;
                let similarity_s_ij = 1.0 / (1.0 + distance);
                alpha_d_sum += causal_trim_relevance(
                    query_index_i,
                    key_index_j,
                    similarity_s_ij,
                    &config.trim,
                );
                relevance_pair_count += 1;
            }
        }
    }

    if relevance_pair_count == 0 {
        0.0
    } else {
        alpha_d_sum / relevance_pair_count as f32
    }
}
