from __future__ import annotations

import torch

from server.compare_restricted_moe_layer_vs_ref import (
    RestrictedRoutingSpec,
    restrict_router_output,
)


def _print_case(name: str, out: dict) -> None:
    print(f"[restrict] case={name}")
    print(f"[restrict] ids=\n{out['selected_expert_ids']}")
    print(f"[restrict] weights=\n{out['selected_weights']}")
    print(f"[restrict] mask=\n{out['selected_mask']}")
    print()


def test_keep_order_and_renorm() -> None:
    router_out = {
        "selected_expert_ids": torch.tensor(
            [
                [7, 3, 9, 2],
                [5, 8, 1, 4],
            ],
            dtype=torch.int64,
        ),
        "selected_weights": torch.tensor(
            [
                [0.40, 0.30, 0.20, 0.10],
                [0.50, 0.20, 0.20, 0.10],
            ],
            dtype=torch.float32,
        ),
    }

    out = restrict_router_output(
        router_out,
        resident_local_expert_ids=[3, 7, 8],
        top_k=2,
        renormalize=True,
    )
    _print_case("keep_order_and_renorm", out)

    expected_ids = torch.tensor(
        [
            [7, 3],
            [8, -1],
        ],
        dtype=torch.int64,
    )
    expected_weights = torch.tensor(
        [
            [0.40 / 0.70, 0.30 / 0.70],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_mask = torch.tensor(
        [
            [True, True],
            [True, False],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(out["selected_expert_ids"], expected_ids)
    assert torch.allclose(out["selected_weights"], expected_weights, atol=1e-6)
    assert torch.equal(out["selected_mask"], expected_mask)


def test_keep_without_renorm() -> None:
    router_out = {
        "selected_expert_ids": torch.tensor(
            [
                [7, 3, 9, 2],
            ],
            dtype=torch.int64,
        ),
        "selected_weights": torch.tensor(
            [
                [0.40, 0.30, 0.20, 0.10],
            ],
            dtype=torch.float32,
        ),
    }

    out = restrict_router_output(
        router_out,
        resident_local_expert_ids=[3, 7],
        top_k=2,
        renormalize=False,
    )
    _print_case("keep_without_renorm", out)

    expected_ids = torch.tensor([[7, 3]], dtype=torch.int64)
    expected_weights = torch.tensor([[0.40, 0.30]], dtype=torch.float32)
    expected_mask = torch.tensor([[True, True]], dtype=torch.bool)

    assert torch.equal(out["selected_expert_ids"], expected_ids)
    assert torch.allclose(out["selected_weights"], expected_weights, atol=1e-6)
    assert torch.equal(out["selected_mask"], expected_mask)


def test_top_k_truncation() -> None:
    router_out = {
        "selected_expert_ids": torch.tensor(
            [
                [7, 3, 8, 2],
            ],
            dtype=torch.int64,
        ),
        "selected_weights": torch.tensor(
            [
                [0.40, 0.30, 0.20, 0.10],
            ],
            dtype=torch.float32,
        ),
    }

    out = restrict_router_output(
        router_out,
        resident_local_expert_ids=[7, 3, 8],
        top_k=2,
        renormalize=True,
    )
    _print_case("top_k_truncation", out)

    expected_ids = torch.tensor([[7, 3]], dtype=torch.int64)
    expected_weights = torch.tensor([[0.40 / 0.70, 0.30 / 0.70]], dtype=torch.float32)
    expected_mask = torch.tensor([[True, True]], dtype=torch.bool)

    assert torch.equal(out["selected_expert_ids"], expected_ids)
    assert torch.allclose(out["selected_weights"], expected_weights, atol=1e-6)
    assert torch.equal(out["selected_mask"], expected_mask)


def test_all_filtered_should_fail() -> None:
    router_out = {
        "selected_expert_ids": torch.tensor(
            [
                [7, 3, 9, 2],
            ],
            dtype=torch.int64,
        ),
        "selected_weights": torch.tensor(
            [
                [0.40, 0.30, 0.20, 0.10],
            ],
            dtype=torch.float32,
        ),
    }

    try:
        restrict_router_output(
            router_out,
            resident_local_expert_ids=[11, 12],
            top_k=2,
            renormalize=True,
        )
    except RuntimeError as exc:
        print("[restrict] case=all_filtered_should_fail")
        print(f"[restrict] got expected error: {exc}")
        print()
        return

    raise AssertionError("expected RuntimeError when all experts are filtered out")


def test_spec_smoke() -> None:
    spec = RestrictedRoutingSpec(
        resident_local_expert_ids=[1, 2, 3],
        top_k=2,
        renormalize=True,
    )
    print("[restrict] spec =", spec)
    print()


def main() -> None:
    test_spec_smoke()
    test_keep_order_and_renorm()
    test_keep_without_renorm()
    test_top_k_truncation()
    test_all_filtered_should_fail()
    print("[restrict] restrict_router_output passed")


if __name__ == "__main__":
    main()
