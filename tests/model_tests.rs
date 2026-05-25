use syncopate_machine::prelude::*;

#[test]
fn estimated_parameter_count_matches_burn_module_count() -> Result<()> {
    let device = Device::default();
    let config = SyncopateModelConfig::tiny_for_tests();
    let model = DefaultSyncopateModel::new(config.clone(), &device)?;

    assert_eq!(config.estimated_parameter_count(), model.parameter_count());
    Ok(())
}

#[test]
fn syncopate_model_forward_has_expected_shape() -> Result<()> {
    let device = Device::default();
    let config = SyncopateModelConfig::tiny_for_tests();
    let model = DefaultSyncopateModel::new(config.clone(), &device)?;
    let action_ids = Tensor::<DefaultAutodiffBackend, 2, Int>::from_data(
        TensorData::new(vec![1i32, 2, 3, 4, 2, 3, 4, 5], [1, config.seq_len]),
        &device,
    );

    let logits = model.forward(action_ids);
    assert_eq!(logits.dims(), [1, config.seq_len, config.vocab_size]);
    Ok(())
}

#[test]
fn syncopate_model_can_train_action_sequences() -> Result<()> {
    let device = Device::default();
    let config = SyncopateModelConfig::tiny_for_tests();
    let mut model = DefaultSyncopateModel::new(config, &device)?;
    let training = ModelTrainingConfig {
        steps: 2,
        batch_size: 2,
        learning_rate: 1e-3,
        min_learning_rate: 1e-4,
        warmup_steps: 1,
        weight_decay: 0.0,
        grad_clip_norm: Some(1.0),
        pad_action_id: 0,
        checkpoint_dir: None,
        checkpoint_interval: 0,
    };

    let report = model.train_action_sequences(
        &[vec![1, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]],
        &training,
        &device,
        |_, _| {},
    )?;
    assert_eq!(report.steps, 2);
    assert!(report.final_loss.is_finite());
    let logits = model.forward_logits(&[1, 2], 0, &device)?;
    assert_eq!(
        logits.dims(),
        [1, model.config().seq_len, model.config().vocab_size]
    );
    Ok(())
}

#[test]
fn syncopate_model_can_save_and_load_parameters() -> Result<()> {
    let device = Device::default();
    let config = SyncopateModelConfig::tiny_for_tests();
    let model = DefaultSyncopateModel::new(config.clone(), &device)?;
    let mut restored = DefaultSyncopateModel::new(config, &device)?;
    let temp = tempfile::tempdir().map_err(|err| syncopate_machine::Error::Io(err.to_string()))?;
    let path = temp.path().join("syncopate");

    model.save_parameters(&path)?;
    restored.load_parameters(&path)?;

    assert_eq!(restored.parameter_count(), model.parameter_count());
    Ok(())
}
