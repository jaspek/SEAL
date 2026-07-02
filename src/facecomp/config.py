"""Central config: resolves machine-specific paths from configs/paths.yaml.
Falls back to configs/paths.example.yaml so the synthetic path runs with zero setup."""
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = REPO_ROOT / "configs"


def _load():
    for name in ("paths.yaml", "paths.example.yaml"):
        p = _CONFIG_DIR / name
        if p.exists():
            with open(p) as f:
                return (yaml.safe_load(f) or {}), p
    return {}, None


_raw, CONFIG_FILE = _load()


def _resolve(p):
    if not p:
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


EMBEDDINGS_DIR = _resolve(_raw.get("embeddings_dir", "outputs/embeddings"))
RESULTS_DIR = _resolve((_raw.get("outputs") or {}).get("results", "outputs/results"))
FIGURES_DIR = _resolve((_raw.get("outputs") or {}).get("figures", "outputs/figures"))
DATA = {k: _resolve(v) for k, v in (_raw.get("data") or {}).items()}
MODELS = {k: (_resolve(v) if v else None) for k, v in (_raw.get("models") or {}).items()}


def emb_path(dataset: str) -> Path:
    return EMBEDDINGS_DIR / f"emb_{dataset}.npz"


def ensure_dirs():
    for d in (EMBEDDINGS_DIR, RESULTS_DIR, FIGURES_DIR):
        if d:
            d.mkdir(parents=True, exist_ok=True)
