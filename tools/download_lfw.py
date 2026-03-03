from pathlib import Path
from sklearn.datasets import fetch_lfw_people
import cv2

def main():
    base = Path(__file__).resolve().parents[1]
    out_dir = base / "data" / "lfw_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LFW dataset (this may take a few minutes)...")
    lfw = fetch_lfw_people(color=True, resize=1.0, funneled=True)

    print("Saving images to disk...")
    for img, target, fname in zip(lfw.images, lfw.target, lfw.target_names[lfw.target]):
        person_dir = out_dir / fname
        person_dir.mkdir(parents=True, exist_ok=True)

        # unique filename
        idx = len(list(person_dir.glob("*.jpg")))
        out_path = person_dir / f"{fname}_{idx:04d}.jpg"

        # convert RGB -> BGR for OpenCV
        img_bgr = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), img_bgr)

    print("Done.")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()