use crate::config::{MultiscreenConfig, TrimConfig};
use crate::error::{Error, Result};
use crate::screen::Screen;
use crate::tile::Tile;

/// Half-open token span `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

impl TokenSpan {
    pub fn new(start: usize, end: usize) -> Result<Self> {
        if start > end {
            return Err(Error::Layout("token span start must be <= end".into()));
        }
        Ok(Self { start, end })
    }

    pub fn len(self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub fn contains(self, index: usize) -> bool {
        self.start <= index && index < self.end
    }
}

/// Fully materialized screen and tile layout for a token sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScreenLayout {
    sequence_len: usize,
    screens: Vec<Screen>,
}

impl ScreenLayout {
    pub fn build(config: &MultiscreenConfig, sequence_len: usize) -> Result<Self> {
        config.validate()?;

        if sequence_len == 0 {
            return Ok(Self {
                sequence_len,
                screens: Vec::new(),
            });
        }

        let mut screens = Vec::new();
        let mut screen_start = 0usize;

        loop {
            if let Some(max_screen_count) = config.screens.max_screen_count
                && screens.len() >= max_screen_count
            {
                break;
            }

            let screen_end = screen_start
                .saturating_add(config.screens.tokens_per_screen)
                .min(sequence_len);
            let screen_span = TokenSpan::new(screen_start, screen_end)?;
            let screen_index = screens.len();
            let tiles = build_tiles(config, screen_index, screen_span)?;
            screens.push(Screen::new(screen_index, screen_span, tiles));

            if screen_end == sequence_len {
                break;
            }
            let Some(next_start) = next_window_start(
                screen_start,
                sequence_len,
                config.screens.tokens_per_screen,
                config.screens.screen_stride_tokens,
                "screen",
            )?
            else {
                break;
            };
            screen_start = next_start;
        }

        Ok(Self {
            sequence_len,
            screens,
        })
    }

    /// Paper sequence length `T` for this layout.
    pub fn sequence_len(&self) -> usize {
        self.sequence_len
    }

    #[deprecated(note = "use sequence_len for paper-aligned naming")]
    pub fn token_count(&self) -> usize {
        self.sequence_len()
    }

    pub fn screens(&self) -> &[Screen] {
        &self.screens
    }

    pub fn screening_tile_count(&self) -> usize {
        self.screens.iter().map(|screen| screen.tiles.len()).sum()
    }

    pub fn screening_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.screens.iter().flat_map(|screen| screen.tiles.iter())
    }

    #[deprecated(note = "use screening_tiles for paper-aligned naming")]
    pub fn tiles(&self) -> impl Iterator<Item = &Tile> {
        self.screening_tiles()
    }
}

fn build_tiles(
    config: &MultiscreenConfig,
    screen_index: usize,
    screen_span: TokenSpan,
) -> Result<Vec<Tile>> {
    if screen_span.is_empty() {
        return Ok(Vec::new());
    }

    let mut tiles = Vec::new();
    let mut tile_start = screen_span.start;
    let screening_grid = &config.tiles.screening_grid;

    loop {
        let tile_end = tile_start
            .saturating_add(config.tiles.tokens_per_tile)
            .min(screen_span.end);
        let span = TokenSpan::new(tile_start, tile_end)?;
        let tile_index = tiles.len();
        let layer_index = (tile_index / screening_grid.head_count) % screening_grid.layer_count;
        let head_index = tile_index % screening_grid.head_count;
        tiles.push(Tile::new(
            tile_index,
            screen_index,
            layer_index,
            head_index,
            span,
        ));

        if tile_end == screen_span.end {
            break;
        }
        let Some(next_start) = next_window_start(
            tile_start,
            screen_span.end,
            config.tiles.tokens_per_tile,
            config.tiles.tile_stride_tokens,
            "tile",
        )?
        else {
            break;
        };
        tile_start = next_start;
    }

    Ok(tiles)
}

fn next_window_start(
    current_start: usize,
    sequence_end: usize,
    window_len: usize,
    stride: usize,
    label: &str,
) -> Result<Option<usize>> {
    let next_start = current_start
        .checked_add(stride)
        .ok_or_else(|| Error::Layout(format!("{label} stride overflowed usize")))?;

    if next_start < sequence_end {
        return Ok(Some(next_start));
    }

    let tail_start = sequence_end.saturating_sub(window_len);
    if tail_start > current_start {
        Ok(Some(tail_start))
    } else {
        Ok(None)
    }
}

/// Trim-and-square relevance gate from the multiscreen experiment.
pub fn trim_and_square(similarity_s_ij: f32, acceptance_sharpness_r: f32) -> f32 {
    let score = 1.0 - acceptance_sharpness_r * (1.0 - similarity_s_ij);
    score.clamp(0.0, 1.0).powi(2)
}

/// Causal softmask for query/key token positions.
pub fn causal_softmask(query_index_i: usize, key_index_j: usize, screening_window_w: f32) -> f32 {
    if screening_window_w <= 0.0 || key_index_j > query_index_i {
        return 0.0;
    }

    let distance = key_index_j as f32 - query_index_i as f32;
    if distance <= -screening_window_w {
        return 0.0;
    }

    ((distance / screening_window_w) * std::f32::consts::PI).cos() * 0.5 + 0.5
}

/// Combined causal trim relevance for a query/key pair.
pub fn causal_trim_relevance(
    query_index_i: usize,
    key_index_j: usize,
    similarity_s_ij: f32,
    trim: &TrimConfig,
) -> f32 {
    trim_and_square(similarity_s_ij, trim.acceptance_sharpness_r)
        * causal_softmask(query_index_i, key_index_j, trim.screening_window_w)
}
