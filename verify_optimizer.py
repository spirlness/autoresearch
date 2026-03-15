import torch

from autoresearch_trainer.optimizer import MuonAdamW


def assert_not_equal(lhs: torch.Tensor, rhs: torch.Tensor, message: str) -> None:
    if torch.equal(lhs, rhs):
        raise AssertionError(message)


def assert_equal(lhs: torch.Tensor, rhs: torch.Tensor, message: str) -> None:
    if not torch.equal(lhs, rhs):
        raise AssertionError(message)


def main() -> None:
    torch.manual_seed(1337)

    p0 = torch.nn.Parameter(torch.randn(4, 4))
    p1 = torch.nn.Parameter(torch.randn(4, 4))

    optimizer = MuonAdamW(
        [
            dict(
                kind="muon",
                params=[p0, p1],
                lr=0.05,
                momentum=0.95,
                ns_steps=1,
                beta2=0.95,
                weight_decay=0.0,
            )
        ],
        optimizer_compile_backend="off",
        compile_mode="default",
    )

    p0_before = p0.detach().clone()
    p1_before = p1.detach().clone()
    p0.grad = torch.randn_like(p0)
    p1.grad = None
    optimizer.step()

    assert_not_equal(
        p0.detach(), p0_before, "Muon should update parameters with gradients."
    )
    assert_equal(
        p1.detach(),
        p1_before,
        "Muon should leave parameters with grad=None untouched.",
    )

    p0_mid = p0.detach().clone()
    p1_mid = p1.detach().clone()
    p0.grad = None
    p1.grad = torch.randn_like(p1)
    optimizer.step()

    assert_equal(
        p0.detach(),
        p0_mid,
        "Muon should keep skipping inactive parameters across later steps.",
    )
    assert_not_equal(
        p1.detach(),
        p1_mid,
        "Muon should update a parameter once it receives gradients again.",
    )

    state = optimizer.state[p0]
    if state["momentum_buffer"].shape != (2, 4, 4):
        raise AssertionError("Muon momentum buffer shape regressed.")
    if state["second_momentum_buffer"].shape != (2, 4, 1):
        raise AssertionError("Muon second-moment buffer shape regressed.")

    print("Muon grad=None handling smoke test passed.")


if __name__ == "__main__":
    main()
