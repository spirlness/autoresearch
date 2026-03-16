import torch
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('.')
from autoresearch_trainer.config import build_runtime_config
from autoresearch_trainer.runner import Trainer
from autoresearch_trainer.model import compute_mfu

class DummyArgs:
    compile_backend = 'off'
    compile_mode = 'default'
    compile_scope = 'model'
    optimizer_compile_backend = 'off'
    experiment_profile = 'mfu50'
    benchmark_steps = 5
    grad_accum_steps = 1
    seed = 1337

args = DummyArgs()
runtime = build_runtime_config(
    args,
    model_compile_backend='off',
    optimizer_compile_backend='off',
    vocab_size=8192,
)

from unittest.mock import patch

def test_mfu():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties") as mock_props, \
         patch("torch.cuda.is_bf16_supported", return_value=True), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("autoresearch_trainer.model.resolve_attention_backend", return_value=("mock_backend", None)):

        # We patch nn.Module.to to return self so method chaining works
        original_to = torch.nn.Module.to
        def mock_to(self, *args, **kwargs):
            return self

        with patch("torch.nn.Module.to", mock_to):
            mock_props.return_value.major = 8
            mock_props.return_value.minor = 0
            mock_props.return_value.multi_processor_count = 108
            mock_props.return_value.clock_rate = 1410000

            trainer = Trainer(runtime)

            print(f"num_params: {trainer.num_params}")
            print(f"flops/tok: {trainer.num_flops_per_token}")
            print(f"peak_flops: {trainer.device_peak_flops}")

            tok_per_sec = 76152
            device_mfu = compute_mfu(trainer.device_peak_flops, trainer.num_flops_per_token, tok_per_sec)
            print(f"Computed MFU for {tok_per_sec} tok/s = {device_mfu:.2f}%")
            assert device_mfu > 0
