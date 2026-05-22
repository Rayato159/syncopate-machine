use syncopate_machine::prelude::*;

#[test]
fn layout_splits_screens_and_tiles() {
    let config = MultiscreenConfig::tiny();
    let layout = ScreenLayout::build(&config, 10).unwrap();

    assert_eq!(layout.sequence_len(), 10);
    assert_eq!(layout.screens().len(), 2);
    assert_eq!(layout.screens()[0].span, TokenSpan { start: 0, end: 8 });
    assert_eq!(layout.screens()[1].span, TokenSpan { start: 4, end: 10 });
    assert!(layout.screening_tile_count() >= 4);
}

#[test]
fn layout_handles_screen_stride_that_jumps_past_tail() {
    let mut config = MultiscreenConfig::tiny();
    config.screens.tokens_per_screen = 4;
    config.screens.screen_stride_tokens = 20;
    config.tiles.tokens_per_tile = 2;
    config.tiles.tile_stride_tokens = 2;

    let layout = ScreenLayout::build(&config, 10).unwrap();
    let spans = layout
        .screens()
        .iter()
        .map(|screen| screen.span)
        .collect::<Vec<_>>();

    assert_eq!(
        spans,
        vec![
            TokenSpan { start: 0, end: 4 },
            TokenSpan { start: 6, end: 10 },
        ]
    );
}

#[test]
fn layout_handles_tile_stride_that_jumps_past_tail() {
    let mut config = MultiscreenConfig::tiny();
    config.screens.tokens_per_screen = 10;
    config.screens.screen_stride_tokens = 10;
    config.tiles.tokens_per_tile = 4;
    config.tiles.tile_stride_tokens = 20;

    let layout = ScreenLayout::build(&config, 10).unwrap();
    let tile_spans = layout
        .screening_tiles()
        .map(|tile| tile.span)
        .collect::<Vec<_>>();

    assert_eq!(
        tile_spans,
        vec![
            TokenSpan { start: 0, end: 4 },
            TokenSpan { start: 6, end: 10 },
        ]
    );
}

#[test]
fn trim_gate_matches_expected_shape() {
    assert_eq!(trim_and_square(1.0, 2.0), 1.0);
    assert_eq!(trim_and_square(0.4, 2.0), 0.0);
    assert!((trim_and_square(0.75, 2.0) - 0.25).abs() < 1e-6);
}

#[test]
fn causal_softmask_blocks_future_and_far_past() {
    assert_eq!(causal_softmask(2, 3, 4.0), 0.0);
    assert_eq!(causal_softmask(4, 0, 4.0), 0.0);
    assert!((causal_softmask(3, 3, 4.0) - 1.0).abs() < 1e-6);
    assert!(causal_softmask(3, 1, 4.0) > 0.0);
}
