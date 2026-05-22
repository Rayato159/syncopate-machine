use syncopate_machine::prelude::*;

#[test]
fn inference_uses_learned_next_token_counts() {
    let mut engine = MultiscreenEngine::new(MultiscreenConfig::tiny()).unwrap();
    engine
        .train(TrainInput::from_token_sequences(vec![
            vec![1, 2, 3],
            vec![1, 2, 4],
            vec![2, 4, 5],
        ]))
        .unwrap();

    let output = engine.infer_tokens(&[1, 2]).unwrap();
    assert_eq!(output.output_token_ids, vec![2, 4]);
    assert!(output.mean_distance_relevance_alpha_d > 0.0);
}

#[test]
fn untrained_engine_can_use_input_token_fallback_tokens() {
    let engine = MultiscreenEngine::new(MultiscreenConfig::tiny()).unwrap();
    let output = engine.infer_tokens(&[7, 8]).unwrap();
    assert_eq!(output.output_token_ids, vec![7, 8]);
}

// ------------------------------------------------------------------
// Weight persistence tests
// ------------------------------------------------------------------

#[test]
fn save_and_load_weights_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.json");

    let config = MultiscreenConfig::tiny();
    let mut engine = MultiscreenEngine::new(config.clone()).unwrap();
    engine
        .train(TrainInput::from_token_sequences(vec![
            vec![1, 2, 3],
            vec![1, 2, 4],
        ]))
        .unwrap();
    engine.save_weights(&path).unwrap();

    let mut engine2 = MultiscreenEngine::new(config).unwrap();
    let report = engine2.load_weights(&path).unwrap();
    assert_eq!(report.training_sequence_count, 2);
    assert_eq!(report.training_token_count, 6);

    // Inference works after loading
    let output = engine2.infer_tokens(&[1, 2]).unwrap();
    // predict_next_token(1) = 2 (only transition), predict_next_token(2) = 3 (tie between 3 and 4,
    // BTreeMap iterates ascending so 3 is encountered first and wins via max_by)
    assert_eq!(output.output_token_ids, vec![2, 3]);
}

#[test]
fn load_weights_rejects_config_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.json");

    let mut engine = MultiscreenEngine::new(MultiscreenConfig::tiny()).unwrap();
    engine
        .train(TrainInput::from_token_sequences(vec![vec![1, 2, 3]]))
        .unwrap();
    engine.save_weights(&path).unwrap();

    // Try to load into an engine with a DIFFERENT config
    let mut wrong_engine = MultiscreenEngine::new(MultiscreenConfig::default()).unwrap();
    let err = wrong_engine.load_weights(&path).unwrap_err();
    assert!(
        err.to_string().contains("config mismatch"),
        "expected config mismatch error, got: {err}"
    );
}

#[test]
fn from_weights_file_creates_working_engine() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.json");

    let config = MultiscreenConfig::tiny();
    let mut engine = MultiscreenEngine::new(config).unwrap();
    engine
        .train(TrainInput::from_token_sequences(vec![
            vec![10, 20, 30],
            vec![10, 20, 40],
        ]))
        .unwrap();
    engine.save_weights(&path).unwrap();

    let loaded = MultiscreenEngine::from_weights_file(&path).unwrap();
    let output = loaded.infer_tokens(&[10, 20]).unwrap();
    // 10→20 is the most common, 20→40 appears twice (30 and 40, but 40 appears once more)
    assert_eq!(output.output_token_ids.len(), 2);
}

#[test]
fn from_weights_file_rejects_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.json");
    std::fs::write(&path, "not json at all").unwrap();

    let err = MultiscreenEngine::from_weights_file(&path).unwrap_err();
    assert!(err.to_string().contains("serialization error"));
}

#[test]
fn load_weights_rejects_missing_file() {
    let mut engine = MultiscreenEngine::new(MultiscreenConfig::tiny()).unwrap();
    let err = engine
        .load_weights("/tmp/this_file_does_not_exist_xyz.json")
        .unwrap_err();
    assert!(err.to_string().contains("I/O error"));
}
