use crate::error::{Error, Result};
use crate::layout::TokenSpan;
use serde::{Deserialize, Serialize};

/// Logical `N_L x N_H` screening grid from the Multiscreen paper.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScreeningGridConfig {
    /// Paper symbol `N_L`: number of residual screening layers represented by
    /// the layout grid.
    #[serde(alias = "rows")]
    pub layer_count: usize,
    /// Paper symbol `N_H`: number of parallel gated screening tiles per layer.
    #[serde(alias = "columns")]
    pub head_count: usize,
}

impl Default for ScreeningGridConfig {
    fn default() -> Self {
        Self {
            layer_count: 2,
            head_count: 2,
        }
    }
}

impl ScreeningGridConfig {
    pub fn validate(&self) -> Result<()> {
        if self.layer_count == 0 {
            return Err(Error::Config(
                "screening_grid layer_count must be greater than zero".into(),
            ));
        }
        if self.head_count == 0 {
            return Err(Error::Config(
                "screening_grid head_count must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    pub fn tile_slots(&self) -> usize {
        self.layer_count * self.head_count
    }

    #[deprecated(note = "use tile_slots for paper-aligned naming")]
    pub fn cells(&self) -> usize {
        self.tile_slots()
    }
}

/// Controls how screens are split into sliding tiles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileConfig {
    #[serde(alias = "tile_size")]
    pub tokens_per_tile: usize,
    #[serde(alias = "tile_stride")]
    pub tile_stride_tokens: usize,
    #[serde(alias = "grid")]
    pub screening_grid: ScreeningGridConfig,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tokens_per_tile: 32,
            tile_stride_tokens: 16,
            screening_grid: ScreeningGridConfig::default(),
        }
    }
}

impl TileConfig {
    pub fn validate(&self) -> Result<()> {
        if self.tokens_per_tile == 0 {
            return Err(Error::Config(
                "tokens_per_tile must be greater than zero".into(),
            ));
        }
        if self.tile_stride_tokens == 0 {
            return Err(Error::Config(
                "tile_stride_tokens must be greater than zero".into(),
            ));
        }
        self.screening_grid.validate()
    }
}

/// A tile over a contiguous token span, mapped to paper-style layer/head slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    pub index: usize,
    pub screen_index: usize,
    pub layer_index: usize,
    pub head_index: usize,
    pub span: TokenSpan,
}

impl Tile {
    pub fn new(
        index: usize,
        screen_index: usize,
        layer_index: usize,
        head_index: usize,
        span: TokenSpan,
    ) -> Self {
        Self {
            index,
            screen_index,
            layer_index,
            head_index,
            span,
        }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }

    pub fn is_empty(&self) -> bool {
        self.span.is_empty()
    }
}
