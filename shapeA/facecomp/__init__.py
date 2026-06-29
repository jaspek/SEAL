"""
facecomp — CPU-first toolkit for "Shape A":
extreme end-to-end face-template compression evaluated as 1:N retrieval at scale.

Everything in this package runs on CPU (numpy + scikit-learn + faiss-cpu).
The GPU-only pieces (2:4 sparsity speed-ups, latency timing, QAT) live in ../network
and are meant for the home RTX-3060 machine.
"""
from . import compress, evaluate, data

__all__ = ["compress", "evaluate", "data"]
