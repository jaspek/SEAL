"""Leave-one-out fusion ablation: when does adding a model help?

Motivated by the pruning sweep's accidental discovery that collapsing MagFace
IMPROVED the fused template on XQLFW: naive concat fusion has no mechanism to
down-weight a weak member. This quantifies each model's marginal contribution
directly from the existing embeddings (no extraction): every single model,
every 2-model combination, and the full 3-model fusion, per dataset -- plus
exact McNemar + Nadeau-Bengio on 'drop model m vs keep all three'.

  python experiments/run_fusion_ablation.py                # all six datasets
  python experiments/run_fusion_ablation.py --datasets xqlfw

Writes outputs/results/fusion_ablation.csv (metrics per combination) and
outputs/results/fusion_ablation_significance.csv (leave-one-out tests;
positive diff = dropping that model helps)."""
import argparse
import itertools
from pathlib import Path

import pandas as pd

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate import significance as S
from facecomp.evaluate import verification as ver

BASELINE = {"lfw": "emb_lfw_bin.npz"}   # official-crop build
DATASETS = ["lfw", "sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp"]
SHORT = {"arcface": "arc", "magface": "mag", "adaface": "ada"}


def combos():
    for r in (1, 2, 3):
        yield from itertools.combinations(E.MODELS, r)


def analyze(dataset):
    path = C.EMBEDDINGS_DIR / BASELINE.get(dataset, f"emb_{dataset}.npz")
    if not path.exists():
        print(f"SKIP {dataset}: {path.name} not found")
        return [], []
    b = E.load_emb(path)
    if b.pair_idx is None or any(m not in b.per_model for m in E.MODELS):
        print(f"SKIP {dataset}: needs pair_idx and all three models")
        return [], []
    labels = b.pair_idx[:, 2]

    print(f"\n=== {dataset} ===")
    metrics, evals = [], {}
    for ms in combos():
        name = "+".join(SHORT[m] for m in ms)
        f = b.fuse(models=ms)
        scores = ver.cosine_scores(f, b.pair_idx)
        evals[ms] = S.kfold_eval(scores, labels)
        acc, std = ver.evaluate_lfw(f, b.pair_idx)
        row = {"dataset": dataset, "combo": name, "n_models": len(ms),
               "acc": round(acc, 6), "acc_std": round(std, 6)}
        row.update(ver.roc_metrics(f, b.pair_idx))
        metrics.append(row)
        print(f"  {name:12s} acc {acc:.4f}  tar@1% {row['tar_far0.01']:.4f}")

    tests = []
    full = tuple(E.MODELS)
    cf, ff = evals[full]
    for m in E.MODELS:
        without = tuple(x for x in E.MODELS if x != m)
        cw, fw = evals[without]
        d_b, d_c, p_mc = S.mcnemar_exact(cw, cf)     # A = without m, B = all three
        t_nb, p_nb = S.nadeau_bengio(fw, ff)
        row = {"dataset": dataset, "dropped": m,
               "acc_without": round(float(cw.mean()), 6),
               "acc_full": round(float(cf.mean()), 6),
               "diff": round(float(cw.mean() - cf.mean()), 6),
               "discordant_without_wins": d_b, "discordant_full_wins": d_c,
               "p_mcnemar": p_mc, "t_nb": round(t_nb, 3), "p_nb": p_nb}
        tests.append(row)
        mark = "**" if (p_mc < 0.05 and p_nb < 0.05) else ("* " if p_mc < 0.05 else "  ")
        print(f"  {mark}drop {SHORT[m]}: {row['acc_without']:.4f} vs full "
              f"{row['acc_full']:.4f}  diff {row['diff']:+.4f}  "
              f"McNemar p={p_mc:.2g} ({d_b}/{d_c})  NB p={p_nb:.2g}")
    return metrics, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default="fusion_ablation.csv")
    args = ap.parse_args()

    all_metrics, all_tests = [], []
    for d in args.datasets:
        m, t = analyze(d)
        all_metrics += m
        all_tests += t

    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(all_metrics).to_csv(out, index=False)
    sig = out.with_name(out.stem + "_significance.csv")
    pd.DataFrame(all_tests).to_csv(sig, index=False)
    print(f"\nwrote {out}\nwrote {sig}  (** = both tests <0.05, * = McNemar only; "
          f"positive diff = dropping the model helps)")


if __name__ == "__main__":
    main()
