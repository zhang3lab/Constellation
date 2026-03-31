import argparse
import subprocess
import sys


def run_compare_case(
    model_root: str,
    layer_id: int,
    batch_size: int,
    seq_len: int,
    start_pos: int,
    dtype: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "server.compare_shallowmla_torch_triton",
        "--model-root",
        model_root,
        "--layer-id",
        str(layer_id),
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--start-pos",
        str(start_pos),
        "--dtype",
        dtype,
    ]
    print("=" * 80)
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_cache_case(
    model_root: str,
    layer_id: int,
    prefill_len: int,
    decode_steps: int,
    dtype: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "server.compare_shallowmla_cache_torch_triton",
        "--model-root",
        model_root,
        "--layer-id",
        str(layer_id),
        "--prefill-len",
        str(prefill_len),
        "--decode-steps",
        str(decode_steps),
        "--dtype",
        dtype,
    ]
    print("=" * 80)
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--include-cache-regression", action="store_true")
    args = parser.parse_args()

    light_cases = [
        {"batch_size": 1, "seq_len": 1, "start_pos": 0},
        {"batch_size": 1, "seq_len": 16, "start_pos": 0},
        {"batch_size": 1, "seq_len": 1, "start_pos": 32},
        {"batch_size": 1, "seq_len": 512, "start_pos": 0},
        {"batch_size": 1, "seq_len": 2048, "start_pos": 0},
        {"batch_size": 1, "seq_len": 1, "start_pos": 2048},
    ]

    for case in light_cases:
        run_compare_case(
            model_root=args.model_root,
            layer_id=args.layer_id,
            batch_size=case["batch_size"],
            seq_len=case["seq_len"],
            start_pos=case["start_pos"],
            dtype=args.dtype,
        )

    if args.include_cache_regression:
        run_cache_case(
            model_root=args.model_root,
            layer_id=args.layer_id,
            prefill_len=1024,
            decode_steps=128,
            dtype=args.dtype,
        )

    print("=" * 80)
    print("ALL SHALLOWMLA REGRESSION CASES PASSED")


if __name__ == "__main__":
    main()
