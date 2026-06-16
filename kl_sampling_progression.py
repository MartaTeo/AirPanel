import argparse
import json
import os
from pathlib import Path

cache_dir = Path("/tmp/airpanel_kl_sampling_mpl")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIMS = [
    "creativity",
    "humour",
    "originality",
    "reliability_trust",
    "intent_to_purchase",
    "overallpreference",
]

DEFAULT_LABELS = ["Bouygues", "Orange", "Mixed", "Neutral"]

DIM_LABELS = {
    "creativity": "Creativity",
    "humour": "Humour",
    "originality": "Originality",
    "reliability_trust": "Reliability / trust",
    "intent_to_purchase": "Intent to purchase",
    "overallpreference": "Overall preference",
}

COLORS = {
    "creativity": "#214E8A",
    "humour": "#F26A21",
    "originality": "#5B8E7D",
    "reliability_trust": "#6C3483",
    "intent_to_purchase": "#B23A48",
    "overallpreference": "#111111",
}


def detect_base_dir():
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path("/Users/martateodoratrales/Desktop/AirPanel"),
    ]
    for candidate in candidates:
        if (candidate / "preference_annotations_extended.csv").exists():
            return candidate
    return Path.cwd()


def parse_list(raw):
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_sample_sizes(raw, total):
    if raw:
        sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        sizes = [20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, total]
    sizes = sorted({n for n in sizes if 1 <= n <= total})
    if total not in sizes:
        sizes.append(total)
    return sizes


def distribution(series, labels, alpha):
    counts = series.value_counts(dropna=False).reindex(labels, fill_value=0).astype(float).to_numpy()
    return (counts + alpha) / (counts.sum() + alpha * len(labels))


def kl_divergence(p, q):
    return float(np.sum(p * np.log(p / q)))


def winner(series, labels):
    counts = series.value_counts(dropna=False).reindex(labels, fill_value=0)
    return str(counts.idxmax())


def brand_winner(series):
    counts = series.value_counts(dropna=False).reindex(["Bouygues", "Orange"], fill_value=0)
    if counts["Bouygues"] == counts["Orange"]:
        return "Tie"
    return str(counts.idxmax())


def brand_margin(series):
    counts = series.value_counts(dropna=False)
    total = max(1, len(series))
    return 100 * (counts.get("Bouygues", 0) - counts.get("Orange", 0)) / total


def run_sampling(df, dims, labels, sample_sizes, reps, alpha, seed):
    rng = np.random.default_rng(seed)
    records = []
    full = {}

    for dim in dims:
        s = df[dim].dropna().astype(str)
        full[dim] = {
            "series": s,
            "probs": distribution(s, labels, alpha),
            "winner": winner(s, labels),
            "brand_winner": brand_winner(s),
            "brand_margin": brand_margin(s),
        }

    for n in sample_sizes:
        n_reps = 1 if n == len(df) else reps
        for rep in range(n_reps):
            idx = np.arange(len(df)) if n == len(df) else rng.choice(len(df), size=n, replace=False)
            sampled = df.iloc[idx]
            for dim in dims:
                s = sampled[dim].dropna().astype(str)
                p_sample = distribution(s, labels, alpha)
                p_full = full[dim]["probs"]
                sample_winner = winner(s, labels)
                sample_brand_winner = brand_winner(s)
                records.append(
                    {
                        "dimension": dim,
                        "dimension_label": DIM_LABELS.get(dim, dim),
                        "sample_size": n,
                        "rep": rep + 1,
                        "kl_sample_to_full": kl_divergence(p_sample, p_full),
                        "kl_full_to_sample": kl_divergence(p_full, p_sample),
                        "sample_winner": sample_winner,
                        "full_winner": full[dim]["winner"],
                        "winner_matches_full": sample_winner == full[dim]["winner"],
                        "sample_brand_winner": sample_brand_winner,
                        "full_brand_winner": full[dim]["brand_winner"],
                        "brand_winner_matches_full": sample_brand_winner == full[dim]["brand_winner"],
                        "sample_brand_margin_pp": brand_margin(s),
                        "full_brand_margin_pp": full[dim]["brand_margin"],
                    }
                )
    return pd.DataFrame(records)


def summarize(replicates):
    q = (
        replicates
        .groupby(["dimension", "dimension_label", "sample_size"], as_index=False)
        .agg(
            mean_kl_sample_to_full=("kl_sample_to_full", "mean"),
            sd_kl_sample_to_full=("kl_sample_to_full", "std"),
            p10_kl_sample_to_full=("kl_sample_to_full", lambda x: x.quantile(0.10)),
            p90_kl_sample_to_full=("kl_sample_to_full", lambda x: x.quantile(0.90)),
            mean_kl_full_to_sample=("kl_full_to_sample", "mean"),
            winner_match_rate=("winner_matches_full", "mean"),
            brand_winner_match_rate=("brand_winner_matches_full", "mean"),
            mean_brand_margin_pp=("sample_brand_margin_pp", "mean"),
            p05_brand_margin_pp=("sample_brand_margin_pp", lambda x: x.quantile(0.05)),
            p95_brand_margin_pp=("sample_brand_margin_pp", lambda x: x.quantile(0.95)),
            full_winner=("full_winner", "first"),
            full_brand_winner=("full_brand_winner", "first"),
            full_brand_margin_pp=("full_brand_margin_pp", "first"),
            reps=("rep", "count"),
        )
    )
    return q.sort_values(["dimension", "sample_size"])


def decision_table(summary, kl_threshold, winner_threshold):
    rows = []
    for dim, part in summary.groupby("dimension", sort=False):
        ok = part[
            (part["mean_kl_sample_to_full"] <= kl_threshold)
            & (part["brand_winner_match_rate"] >= winner_threshold)
        ].sort_values("sample_size")
        chosen = ok.iloc[0] if len(ok) else part.sort_values("sample_size").iloc[-1]
        rows.append(
            {
                "dimension": dim,
                "dimension_label": chosen["dimension_label"],
                "n_required": int(chosen["sample_size"]),
                "full_brand_winner": chosen["full_brand_winner"],
                "full_brand_margin_pp": round(float(chosen["full_brand_margin_pp"]), 3),
                "mean_kl_sample_to_full": round(float(chosen["mean_kl_sample_to_full"]), 6),
                "brand_winner_match_rate": round(float(chosen["brand_winner_match_rate"]), 4),
                "criterion_met": bool(len(ok)),
            }
        )
    return pd.DataFrame(rows)


def plot_progression(summary, out_path):
    dims = summary["dimension"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for dim in dims:
        part = summary[summary["dimension"] == dim].sort_values("sample_size")
        label = part["dimension_label"].iloc[0]
        color = COLORS.get(dim, None)
        axes[0].plot(part["sample_size"], part["mean_kl_sample_to_full"], marker="o", linewidth=2, label=label, color=color)
        axes[0].fill_between(part["sample_size"], part["p10_kl_sample_to_full"], part["p90_kl_sample_to_full"], color=color, alpha=0.10)
        axes[1].plot(part["sample_size"], part["brand_winner_match_rate"], marker="o", linewidth=2, label=label, color=color)

    axes[0].set_title("KL convergence of sampled preference distributions")
    axes[0].set_ylabel("Mean KL(sample || full)")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_ylabel("Brand winner stability")
    axes[1].set_xlabel("Number of sampled responses")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = detect_base_dir()
    parser = argparse.ArgumentParser(description="Bootstrap KL progression for AirPanel preference annotations.")
    parser.add_argument("--input", default=str(base_dir / "preference_annotations_extended.csv"))
    parser.add_argument("--output_dir", default=str(base_dir / "kl_sampling_progression_outputs"))
    parser.add_argument("--dims", default=",".join(DEFAULT_DIMS))
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--sample_sizes", default="")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl_threshold", type=float, default=0.01)
    parser.add_argument("--winner_threshold", type=float, default=0.95)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    dims = [d for d in parse_list(args.dims) if d in df.columns]
    labels = parse_list(args.labels)
    sample_sizes = parse_sample_sizes(args.sample_sizes, len(df))

    replicates = run_sampling(df, dims, labels, sample_sizes, args.reps, args.alpha, args.seed)
    summary = summarize(replicates)
    decisions = decision_table(summary, args.kl_threshold, args.winner_threshold)

    replicates_path = output_dir / "kl_sampling_progression_replicates.csv"
    summary_path = output_dir / "kl_sampling_progression_summary.csv"
    decisions_path = output_dir / "kl_sampling_progression_decision_table.csv"
    plot_path = output_dir / "kl_sampling_progression_plot.png"
    meta_path = output_dir / "kl_sampling_progression_run.json"

    replicates.to_csv(replicates_path, index=False)
    summary.to_csv(summary_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    plot_progression(summary, plot_path)
    meta_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "dims": dims,
                "labels": labels,
                "sample_sizes": sample_sizes,
                "reps": args.reps,
                "alpha": args.alpha,
                "seed": args.seed,
                "kl_threshold": args.kl_threshold,
                "winner_threshold": args.winner_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved decision table: {decisions_path}")
    print("")
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
