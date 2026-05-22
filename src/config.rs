use crate::error::{Error, Result};
use crate::screen::ScreenConfig;
use crate::tile::TileConfig;
use serde::{Deserialize, Serialize};

/// Multi-screen scoring parameters.
///
/// `acceptance_sharpness_r` matches the trim-and-square gate from the experiment:
/// `square(clamp(1 - acceptance_sharpness_r * (1 - similarity), 0, 1))`.
///
/// `screening_window_w` controls the causal softmask width in token positions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrimConfig {
    /// Paper symbol `r`; larger values narrow the acceptance width `1 / r`.
    #[serde(alias = "radius")]
    pub acceptance_sharpness_r: f32,
    /// Paper symbol `w`; causal distance window used by the softmask.
    #[serde(alias = "window")]
    pub screening_window_w: f32,
}

impl Default for TrimConfig {
    fn default() -> Self {
        Self {
            acceptance_sharpness_r: 2.0,
            screening_window_w: 32.0,
        }
    }
}

impl TrimConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.acceptance_sharpness_r.is_finite() || self.acceptance_sharpness_r <= 0.0 {
            return Err(Error::Config(
                "trim acceptance_sharpness_r must be positive and finite".into(),
            ));
        }
        if !self.screening_window_w.is_finite() || self.screening_window_w <= 0.0 {
            return Err(Error::Config(
                "trim screening_window_w must be positive and finite".into(),
            ));
        }
        Ok(())
    }
}

/// Controls deterministic inference behavior for the lightweight transition
/// engine (`MultiscreenEngine`).
///
/// This is distinct from [`ModelInferenceConfig`](crate::ModelInferenceConfig)
/// which controls the neural `SyncopateModel` inference path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum number of output tokens to produce. `None` means output length
    /// equals input length.
    #[serde(alias = "max_output_tokens")]
    pub max_inference_tokens: Option<usize>,
    /// When `true` and no trained transition exists for a token, fall back to
    /// echoing the input token.
    #[serde(alias = "fallback_to_input")]
    pub use_input_token_fallback: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_inference_tokens: None,
            use_input_token_fallback: true,
        }
    }
}

impl InferenceConfig {
    pub fn validate(&self) -> Result<()> {
        if matches!(self.max_inference_tokens, Some(0)) {
            return Err(Error::Config(
                "max_inference_tokens must be omitted or greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Full engine configuration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiscreenConfig {
    pub screens: ScreenConfig,
    pub tiles: TileConfig,
    pub trim: TrimConfig,
    pub inference: InferenceConfig,
}

impl MultiscreenConfig {
    /// Validates all nested configuration.
    pub fn validate(&self) -> Result<()> {
        self.screens.validate()?;
        self.tiles.validate()?;
        self.trim.validate()?;
        self.inference.validate()?;
        Ok(())
    }

    /// Returns a compact config useful for examples and fast tests.
    pub fn tiny() -> Self {
        Self {
            screens: ScreenConfig {
                tokens_per_screen: 8,
                screen_stride_tokens: 4,
                max_screen_count: None,
            },
            tiles: TileConfig {
                tokens_per_tile: 4,
                tile_stride_tokens: 2,
                screening_grid: crate::tile::ScreeningGridConfig {
                    layer_count: 1,
                    head_count: 2,
                },
            },
            trim: TrimConfig {
                acceptance_sharpness_r: 2.0,
                screening_window_w: 4.0,
            },
            inference: InferenceConfig::default(),
        }
    }
}
