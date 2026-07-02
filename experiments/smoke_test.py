"""End-to-end validation on synthetic data - needs no datasets or models."""
from facecomp.data import emb as E
from facecomp.evaluate import rate_distortion as RD


def main():
    b = E.make_synthetic_bundle(n_ids=100, per_id=6, dim=512, seed=1)
    fused = b.fuse(method="concat")
    print(f"synthetic: N={b.N}, fused_dim={fused.shape[1]}, ids={len(set(b.labels))}")
    df = RD.rate_distortion(fused, labels=b.labels, num_distractors=2000, seed=1).sort_values("bits")
    print(df.to_string(index=False))
    assert len(df) > 0 and (df["bits"] > 0).all()
    assert df["rank1"].max() > 0.5, "synthetic 1:N looks degenerate; check make_synthetic scale"
    print("\nOK: harness runs end-to-end on synthetic data.")


if __name__ == "__main__":
    main()
