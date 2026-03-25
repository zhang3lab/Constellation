import argparse
from pathlib import Path
from safetensors import safe_open

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--layer-id", type=int, default=0)
    args = ap.parse_args()

    prefix = f"model.layers.{args.layer_id}."
    shard_paths = sorted(Path(args.model_root).rglob("*.safetensors"))

    names = []
    for shard in shard_paths:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    names.append((k, str(shard), tuple(f.get_tensor(k).shape), str(f.get_tensor(k).dtype)))

    names.sort(key=lambda x: x[0])
    for k, shard, shape, dtype in names:
        print(k, shape, dtype, shard)

if __name__ == "__main__":
    main()
