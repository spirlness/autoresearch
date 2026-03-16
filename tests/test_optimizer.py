import torch
from unittest.mock import patch

from autoresearch_trainer.optimizer import MuonAdamW


def test_muon_adamw_initialization():
    # Create some dummy parameters
    p1 = torch.nn.Parameter(torch.randn(10, 10))
    p2 = torch.nn.Parameter(torch.randn(10))
    p3 = torch.nn.Parameter(torch.randn(5, 5))

    param_groups = [
        dict(
            kind="muon",
            params=[p1, p3],
            lr=0.01,
            momentum=0.9,
            ns_steps=5,
            beta2=0.95,
            weight_decay=0.01,
        ),
        dict(
            kind="adamw",
            params=[p2],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
    ]

    with patch('autoresearch_trainer.optimizer.maybe_compile_function', side_effect=lambda f, **kwargs: f):
        optimizer = MuonAdamW(
            param_groups,
            optimizer_compile_backend="off",
            compile_mode="default"
        )

        assert len(optimizer.param_groups) == 2

        adamw_group = next(g for g in optimizer.param_groups if g["kind"] == "adamw")
        assert len(adamw_group["params"]) == 1
        assert adamw_group["lr"] == 0.001

        muon_group = next(g for g in optimizer.param_groups if g["kind"] == "muon")
        assert len(muon_group["params"]) == 2
        assert muon_group["lr"] == 0.01

def test_muon_adamw_step_adamw():
    p = torch.nn.Parameter(torch.ones(10))
    p.grad = torch.ones(10) * 0.1

    param_groups = [
        dict(
            kind="adamw",
            params=[p],
            lr=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
    ]

    with patch('autoresearch_trainer.optimizer.maybe_compile_function', side_effect=lambda f, **kwargs: f):
        optimizer = MuonAdamW(
            param_groups,
            optimizer_compile_backend="off",
            compile_mode="default"
        )

        initial_p = p.clone()
        optimizer.step()

        assert not torch.allclose(p, initial_p)
        assert optimizer.state[p]["step"] == 1
        assert "exp_avg" in optimizer.state[p]
        assert "exp_avg_sq" in optimizer.state[p]

def test_muon_adamw_step_muon():
    p = torch.nn.Parameter(torch.ones(10, 10))
    p.grad = torch.ones(10, 10) * 0.1

    param_groups = [
        dict(
            kind="muon",
            params=[p],
            lr=0.1,
            momentum=0.9,
            ns_steps=5,
            beta2=0.95,
            weight_decay=0.0,
        )
    ]

    with patch('autoresearch_trainer.optimizer.maybe_compile_function', side_effect=lambda f, **kwargs: f):
        optimizer = MuonAdamW(
            param_groups,
            optimizer_compile_backend="off",
            compile_mode="default"
        )

        initial_p = p.clone()
        optimizer.step()

        assert not torch.allclose(p, initial_p)
        assert "momentum_buffer" in optimizer.state[p]
        assert "second_momentum_buffer" in optimizer.state[p]
