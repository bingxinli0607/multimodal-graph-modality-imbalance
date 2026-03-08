import subprocess
import sys
import re
from pathlib import Path


# ============================================================
# 配置区：你以后主要改这里
# ============================================================

EPOCHS = 200
SEEDS = [1, 2, 3, 4, 5]

SETTINGS = [
    {
        "name": "baseline",
        "miss_b": 0.0,
        "moddrop": 0.0,
    },
    {
        "name": "missing-B",
        "miss_b": 0.5,
        "moddrop": 0.0,
    },
    {
        "name": "+moddrop",
        "miss_b": 0.5,
        "moddrop": 0.5,
    },
]


# ============================================================
# 路径设置
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "src" / "train_imbalance.py"
RESULT_PATH = REPO_ROOT / "results" / "day2_modality_imbalance_auto.md"


# ============================================================
# 正则：从 train_imbalance.py 的输出里提取 best_test
# 匹配形如：
# Summary: miss_b=0.50, moddrop=0.50, best_test=0.8090
# ============================================================

SUMMARY_PATTERN = re.compile(
    r"Summary:\s*miss_b=(?P<miss_b>[0-9.]+),\s*moddrop=(?P<moddrop>[0-9.]+),\s*best_test=(?P<best_test>[0-9.]+)"
)


def run_one(seed: int, miss_b: float, moddrop: float, epochs: int) -> float:
    """
    跑一次 train_imbalance.py，并返回 best_test。
    """
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--epochs", str(epochs),
        "--miss_b", str(miss_b),
        "--moddrop", str(moddrop),
        "--seed", str(seed),
    ]

    print("=" * 70)
    print("Running:", " ".join(cmd))
    print("=" * 70)

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(
            f"Experiment failed: seed={seed}, miss_b={miss_b}, moddrop={moddrop}"
        )

    match = SUMMARY_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError(
            "Could not parse Summary line from output.\n"
            "Please check train_imbalance.py prints:\n"
            "Summary: miss_b=..., moddrop=..., best_test=..."
        )

    best_test = float(match.group("best_test"))
    return best_test


def format_markdown_table(rows):
    """
    把实验结果格式化成 markdown 表格。
    """
    header = (
        "# Day2 Modality Imbalance (Missing View-B Edges)\n\n"
        "Dataset: Cora (Planetoid)  \n"
        "View-A: citation graph  \n"
        "View-B: kNN graph  \n"
        f"epoch: {EPOCHS}  \n"
        f"seeds: {', '.join(map(str, SEEDS))}  \n\n"
        "| setting | miss_b (eval) | moddrop (train) | "
        + " | ".join([f"seed{s}" for s in SEEDS])
        + " | avg |\n"
        "|---|---:|---:|"
        + "|".join(["---:"] * len(SEEDS))
        + "|---:|\n"
    )

    lines = [header]
    for row in rows:
        seed_str = " | ".join([f"{x:.4f}" for x in row["scores"]])
        avg_str = f"{row['avg']:.4f}"
        line = (
            f"| {row['name']} | {row['miss_b']:.2f} | {row['moddrop']:.2f} | "
            f"{seed_str} | {avg_str} |\n"
        )
        lines.append(line)

    lines.append("\n")
    lines.append("## Observation\n\n")
    lines.append(
        "This table reports the initial benchmark results under the fixed evaluation setting. "
        "More seeds, more epochs, and multiple missing rates are recommended for a more reliable conclusion.\n"
    )
    return "".join(lines)


def main():
    rows = []

    for setting in SETTINGS:
        scores = []
        print(
            f"\n\n>>> Setting: {setting['name']} | miss_b={setting['miss_b']} | moddrop={setting['moddrop']}\n"
        )

        for seed in SEEDS:
            score = run_one(
                seed=seed,
                miss_b=setting["miss_b"],
                moddrop=setting["moddrop"],
                epochs=EPOCHS,
            )
            print(
                f"[RESULT] setting={setting['name']}, seed={seed}, best_test={score:.4f}"
            )
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        rows.append(
            {
                "name": setting["name"],
                "miss_b": setting["miss_b"],
                "moddrop": setting["moddrop"],
                "scores": scores,
                "avg": avg_score,
            }
        )

    md = format_markdown_table(rows)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(md, encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Saved markdown result to: {RESULT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()