from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():

    base = Path(__file__).resolve().parents[1]
    csv_path = base / "results" / "arcface_leakage_0_100.csv"

    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("percent")

    print(df)

    out_dir = base / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) EER curve
    # --------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(df["percent"], df["eer"], marker="o")
    plt.xlabel("Encrypted Payload (%)")
    plt.ylabel("Equal Error Rate (EER)")
    plt.title("Recognition Degradation under Selective JPEG2000 Encryption")
    plt.grid(True)

    plt.savefig(out_dir / "eer_vs_encryption.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # 2) TAR @ FAR = 1%
    # --------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(df["percent"], df["tar@far=1%"], marker="o")
    plt.xlabel("Encrypted Payload (%)")
    plt.ylabel("TAR @ FAR = 1%")
    plt.title("Verification Performance (FAR = 1%)")
    plt.grid(True)

    plt.savefig(out_dir / "tar_far1_vs_encryption.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # 3) TAR @ FAR = 0.1%
    # --------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(df["percent"], df["tar@far=0.1%"], marker="o")
    plt.xlabel("Encrypted Payload (%)")
    plt.ylabel("TAR @ FAR = 0.1%")
    plt.title("Strict Verification Performance (FAR = 0.1%)")
    plt.grid(True)

    plt.savefig(out_dir / "tar_far01_vs_encryption.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # 4) Combined plot (BEST FOR THESIS)
    # --------------------------------------------------
    plt.figure(figsize=(8,5))

    plt.plot(df["percent"], df["eer"], marker="o", label="EER")
    plt.plot(df["percent"], df["tar@far=1%"], marker="s", label="TAR @ FAR=1%")
    plt.plot(df["percent"], df["tar@far=0.1%"], marker="^", label="TAR @ FAR=0.1%")

    plt.xlabel("Encrypted Payload (%)")
    plt.ylabel("Performance")
    plt.title("Face Recognition vs Selective Encryption Strength")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_dir / "combined_performance.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # 5) Biometric Leakage vs Structural Similarity
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(df["ssim_mean"], df["tar@far=0.1%"], marker="o")
    plt.xlabel("SSIM (mean)")
    plt.ylabel("TAR @ FAR = 0.1%")
    plt.title("Biometric Leakage vs Structural Similarity")
    plt.grid(True)
    plt.savefig(out_dir / "ssim_vs_tar01.png", dpi=300)
    plt.close()

    print("\nPlots saved to:", out_dir)


if __name__ == "__main__":
    main()