use syncopate_machine::prelude::*;

#[test]
fn parameter_budget_presets_validate_and_scale_up() -> Result<()> {
    let vocab_size = 8192;
    let seq_len = 96;
    let mut previous_count = 0;

    for budget in SyncopateParameterBudget::ALL {
        let config = SyncopateModelConfig::for_parameter_budget(budget, vocab_size, seq_len);
        config.validate()?;

        let count = config.estimated_parameter_count();
        let target = budget.target_parameter_count();
        assert!(
            count >= target * 3 / 4 && count <= target * 5 / 4,
            "{} preset estimated {} params, target {}",
            budget.label(),
            count,
            target
        );
        assert!(
            count > previous_count,
            "{} preset did not increase parameter count",
            budget.label()
        );
        previous_count = count;
    }

    Ok(())
}

#[test]
fn estimated_parameter_count_matches_burn_module_count() -> Result<()> {
    let device = Device::default();
    let config = SyncopateModelConfig::tiny_for_tests();
    let model = DefaultSyncopateModel::new(config.clone(), &device)?;

    assert_eq!(config.estimated_parameter_count(), model.parameter_count());
    Ok(())
}

#[test]
fn preset_10m_uses_tiny_llama_style_dimensions() {
    let config = SyncopateModelConfig::paper_10m(8192, 96);

    assert_eq!(config, SyncopateModelConfig::preset_10m(8192, 96));
    assert_eq!(config.layers, 10);
    assert_eq!(config.d_model, 256);
    assert_eq!(config.attention_heads, 8);
    assert_eq!(config.kv_heads, 2);
    assert_eq!(config.intermediate_size, 704);
}

#[test]
fn syncopate_model_forward_has_expected_shape() -> Result<()> {
    let device = Device::default();
    let config = SyncopateModelConfig::tiny_for_tests();
    let model = DefaultSyncopateModel::new(config.clone(), &device)?;
    let tokens = Tensor::<DefaultAutodiffBackend, 2, Int>::from_data(
        TensorData::new(vec![1i32, 2, 3, 4, 2, 3, 4, 5], [1, config.seq_len]),
        &device,
    );

    let logits = model.forward(tokens);
    assert_eq!(logits.dims(), [1, config.seq_len, config.vocab_size]);
    Ok(())
}

#[test]
fn syncopate_model_can_train_and_infer_tokens() -> Result<()> {
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
        pad_token_id: 0,
        checkpoint_dir: None,
        checkpoint_interval: 0,
    };

    let report = model.train_token_sequences(
        &[vec![1, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]],
        &training,
        &device,
        |_, _| {},
    )?;
    assert_eq!(report.steps, 2);
    assert!(report.final_loss.is_finite());

    let output = model.infer_tokens(
        &[1, 2],
        &ModelInferenceConfig {
            max_new_tokens: 2,
            pad_token_id: 0,
        },
        &device,
    )?;
    assert_eq!(output.token_ids.len(), 4);
    Ok(())
}

#[test]
fn syncopate_model_supports_higher_order_attention() -> Result<()> {
    let device = Device::default();
    let config =
        SyncopateModelConfig::tiny_for_tests().with_attention_kernel(AttentionKernel::HigherOrder);
    let model = DefaultSyncopateModel::new(config.clone(), &device)?;
    let tokens = Tensor::<DefaultAutodiffBackend, 2, Int>::from_data(
        TensorData::new(vec![1i32, 2, 3, 4, 2, 3, 4, 5], [1, config.seq_len]),
        &device,
    );

    let logits = model.forward(tokens);
    assert_eq!(logits.dims(), [1, config.seq_len, config.vocab_size]);
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
