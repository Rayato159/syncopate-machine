use crate::error::{Error, Result};
use crate::layout::TokenSpan;
use crate::tile::Tile;
use serde::{Deserialize, Serialize};

/// Controls how a token sequence is split into screens.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScreenConfig {
    #[serde(alias = "screen_size")]
    pub tokens_per_screen: usize,
    #[serde(alias = "screen_stride")]
    pub screen_stride_tokens: usize,
    #[serde(alias = "max_screens", alias = "max_screen_layout_count")]
    pub max_screen_count: Option<usize>,
}

impl Default for ScreenConfig {
    fn default() -> Self {
        Self {
            tokens_per_screen: 128,
            screen_stride_tokens: 64,
            max_screen_count: None,
        }
    }
}

impl ScreenConfig {
    pub fn validate(&self) -> Result<()> {
        if self.tokens_per_screen == 0 {
            return Err(Error::Config(
                "tokens_per_screen must be greater than zero".into(),
            ));
        }
        if self.screen_stride_tokens == 0 {
            return Err(Error::Config(
                "screen_stride_tokens must be greater than zero".into(),
            ));
        }
        if matches!(self.max_screen_count, Some(0)) {
            return Err(Error::Config(
                "max_screen_count must be omitted or greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// A screen over a contiguous token span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Screen {
    pub index: usize,
    pub span: TokenSpan,
    pub tiles: Vec<Tile>,
}

impl Screen {
    pub fn new(index: usize, span: TokenSpan, tiles: Vec<Tile>) -> Self {
        Self { index, span, tiles }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }

    pub fn is_empty(&self) -> bool {
        self.span.is_empty()
    }
}
