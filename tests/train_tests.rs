use syncopate_machine::prelude::*;

#[test]
fn train_from_token_sequences_builds_report() {
    let mut engine = MultiscreenEngine::new(MultiscreenConfig::tiny()).unwrap();
    let report = engine
        .train(TrainInput::from_token_sequences(vec![
            vec![1, 2, 3],
            vec![1, 2, 4],
        ]))
        .unwrap();

    assert_eq!(report.training_sequence_count, 2);
    assert_eq!(report.training_token_count, 6);
    assert_eq!(report.observed_vocab_size, 4);
    assert!(report.screen_layout_count > 0);
    assert!(report.screening_tile_count > 0);
}
