use syncopate_machine::prelude::*;

#[test]
fn default_config_is_valid() {
    MultiscreenConfig::default().validate().unwrap();
}

#[test]
fn rejects_zero_tokens_per_screen() {
    let mut config = MultiscreenConfig::default();
    config.screens.tokens_per_screen = 0;
    assert!(config.validate().is_err());
}

#[test]
fn rejects_zero_tile_stride_tokens() {
    let mut config = MultiscreenConfig::default();
    config.tiles.tile_stride_tokens = 0;
    assert!(config.validate().is_err());
}
