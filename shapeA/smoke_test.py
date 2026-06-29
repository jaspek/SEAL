#!/usr/bin/env python3
"""
End-to-end smoke test on SYNTHETIC data (no models, no datasets needed).
Validates the whole harness in a few seconds on CPU. Run this first.

  python smoke_test.py
"""
import numpy as np

from facecomp import compress, evaluate, data


def main():
    # 400 identities x 4 images + 5000 pure distractors, 512-d
    X, y = data.make_synthetic(n_ids=400, imgs_per_id=4, dim=512,
                               noise=0.7, n_distractor=0, seed=1)
    split = data.build_gallery_probe(y, seed=1)
    gX, gids = X[split["gallery_idx"]], y[split["gallery_idx"]]
    pX, pids = X[split["probe_idx"]], y[split["probe_idx"]]

    # scale gallery with synthetic distractors
    dX, _ = data.make_synthetic(n_ids=5000, imgs_per_id=1, dim=512, seed=99)
    gX, gids = data.append_distractors(gX, gids, dX)

    print(f"gallery={gX.shape[0]}  probes={pX.shape[0]}  dim={X.shape[1]}\n")
    print(f"{'method':14s} {'bits':>6s} {'R1':>7s} {'R5':>7s} {'mAP':>7s} "
          f"{'EER':>7s} {'TAR1e-2':>8s}")

    results = {}
    for comp in compress.default_zoo(k=128):
        try:
            comp.fit(gX)
        except Exception as e:
            print(f"[skip] {comp.name}: {e}")
            continue
        r = evaluate.evaluate_compressor(comp, gX, gids, pX, pids,
                                         all_X=X, all_ids=y, n_pairs=5000)
        results[comp.name] = r
        print(f"{comp.name:14s} {r['bits_per_vec']:6d} {r['rank1']:7.4f} "
              f"{r['rank5']:7.4f} {r['map']:7.4f} {r.get('eer', float('nan')):7.4f} "
              f"{r.get('tar@far=0.01', float('nan')):8.4f}")

    # sanity checks
    assert results["fp32"]["rank1"] > 0.5, "fp32 retrieval looks broken"
    assert results["binary"]["rank1"] <= results["fp32"]["rank1"] + 1e-6
    assert results["fp16"]["rank1"] >= results["fp32"]["rank1"] - 0.02
    print("\nOK: harness runs end-to-end and metrics are ordered sanely.")


if __name__ == "__main__":
    main()
