import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bench_mlp_clean.csv")

# 1) large baseline: latency vs batch
large = df[
    (df["hidden"] == 7168) &
    (df["inter"] == 2048) &
    (df["k_chunk"] == 1024)
].copy()

for fmt in ["E4M3", "E5M2"]:
    sub = large[large["fmt"] == fmt].sort_values("batch")
    plt.plot(sub["batch"], sub["mean_ms"], marker="o", label=fmt)

plt.xlabel("batch")
plt.ylabel("mean latency (ms)")
plt.title("MLP latency vs batch (7168x2048, k_chunk=1024)")
plt.legend()
plt.xscale("log", base=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_latency_vs_batch.png", dpi=200)
plt.close()

# 2) large baseline: throughput vs batch
for fmt in ["E4M3", "E5M2"]:
    sub = large[large["fmt"] == fmt].sort_values("batch")
    plt.plot(sub["batch"], sub["throughput_tok_s"], marker="o", label=fmt)

plt.xlabel("batch")
plt.ylabel("throughput (tokens/s)")
plt.title("MLP throughput vs batch (7168x2048, k_chunk=1024)")
plt.legend()
plt.xscale("log", base=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_throughput_vs_batch.png", dpi=200)
plt.close()

# 3) k_chunk scan: latency vs k_chunk for E4M3 large shape
scan = df[
    (df["hidden"] == 7168) &
    (df["inter"] == 2048) &
    (df["fmt"] == "E4M3") &
    (df["batch"].isin([1, 4, 16])) &
    (df["k_chunk"].isin([256, 512, 1024]))
].copy()

for b in [1, 4, 16]:
    sub = scan[scan["batch"] == b].sort_values("k_chunk")
    plt.plot(sub["k_chunk"], sub["mean_ms"], marker="o", label=f"batch={b}")

plt.xlabel("k_chunk")
plt.ylabel("mean latency (ms)")
plt.title("MLP latency vs k_chunk (7168x2048, E4M3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_latency_vs_kchunk.png", dpi=200)
plt.close()

print("Saved: mlp_latency_vs_batch.png, mlp_throughput_vs_batch.png, mlp_latency_vs_kchunk.png")
