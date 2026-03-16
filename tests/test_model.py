import torch
from unittest.mock import MagicMock

from autoresearch_trainer.model import (
    estimate_device_peak_flops,
    compute_mfu,
    target_tok_per_sec_for_mfu,
    GPTConfig,
    build_model_config,
    norm,
    has_ve,
    GPT
)


def test_estimate_device_peak_flops():
    mock_props = MagicMock()
    mock_props.major = 8
    mock_props.minor = 0
    mock_props.multi_processor_count = 108
    mock_props.clock_rate = 1410000

    peak_flops = estimate_device_peak_flops(mock_props)
    expected_flops = 108 * 2048 * 1410000 * 1000
    assert peak_flops == expected_flops


def test_estimate_device_peak_flops_unknown_capability():
    mock_props = MagicMock()
    mock_props.major = 6
    mock_props.minor = 1
    mock_props.multi_processor_count = 108
    mock_props.clock_rate = 1410000

    peak_flops = estimate_device_peak_flops(mock_props)
    assert peak_flops is None


def test_compute_mfu():
    peak_flops = 1000.0
    num_flops_per_token = 10.0
    tok_per_sec = 50.0

    mfu = compute_mfu(peak_flops, num_flops_per_token, tok_per_sec)
    expected_mfu = 100 * 10.0 * 50.0 / 1000.0
    assert mfu == expected_mfu


def test_compute_mfu_edge_cases():
    assert compute_mfu(None, 10, 50) is None
    assert compute_mfu(1000, 10, None) is None
    assert compute_mfu(0, 10, 50) is None
    assert compute_mfu(1000, 10, 0) is None


def test_target_tok_per_sec_for_mfu():
    peak_flops = 1000.0
    num_flops_per_token = 10.0
    target_mfu_percent = 50.0

    tok_per_sec = target_tok_per_sec_for_mfu(peak_flops, num_flops_per_token, target_mfu_percent)
    expected_tok_per_sec = 1000.0 * (50.0 / 100.0) / 10.0
    assert tok_per_sec == expected_tok_per_sec


def test_target_tok_per_sec_for_mfu_edge_cases():
    assert target_tok_per_sec_for_mfu(None, 10, 50) is None
    assert target_tok_per_sec_for_mfu(0, 10, 50) is None
    assert target_tok_per_sec_for_mfu(1000, 0, 50) is None
    assert target_tok_per_sec_for_mfu(1000, 10, 0) is None


def test_build_model_config():
    config = build_model_config(
        depth=12,
        max_seq_len=2048,
        vocab_size=32000,
        aspect_ratio=64.0,
        head_dim=64,
        window_pattern="SSSL",
        activation_checkpoint="none",
        ve_gate_channels=32,
        softcap=15.0
    )
    assert isinstance(config, GPTConfig)
    assert config.n_layer == 12
    assert config.sequence_len == 2048
    assert config.vocab_size == 32000
    assert config.n_head == 12  # (12 * 64) // 64
    assert config.n_embd == 768
    assert config.n_kv_head == 12


def test_norm():
    x = torch.randn(2, 4, 8)
    y = norm(x)
    assert y.shape == x.shape


def test_has_ve():
    assert not has_ve(0, 12)
    assert has_ve(1, 12)
    assert not has_ve(2, 12)
    assert not has_ve(10, 12)
    assert has_ve(11, 12)


def mock_attention_op(q, k, v, causal, window_size):
    # Dummy attention op returning proper shape
    return torch.zeros_like(q)


def test_gpt_init_and_estimate_flops():
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="LL",
        activation_checkpoint="none"
    )
    model = GPT(config, attention_op=mock_attention_op)
    model.init_weights()
    flops = model.estimate_flops()
    assert isinstance(flops, int)
    assert flops > 0

    scaling_params = model.num_scaling_params()
    assert "total" in scaling_params
    assert scaling_params["total"] > 0
