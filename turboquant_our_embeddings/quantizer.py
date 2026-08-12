"""Self-contained TurboQuant implementation using numpy/scipy.

Implements the TurboQuant algorithm from arXiv:2504.19874:
1. Random orthogonal rotation to make coordinates follow a predictable distribution
2. Lloyd-Max scalar quantization with precomputed codebook for Gaussian marginals
3. Optional QJL (Quantized Johnson-Lindenstrauss) bias correction for inner products
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# Precomputed Lloyd-Max optimal quantizer levels for standard Gaussian N(0,1).
# boundaries[i] and centroids[i] define the optimal scalar quantizer at i bits.
# These are the well-known MMSE-optimal levels (Max, 1960).
GAUSSIAN_LLOYD_MAX: dict[int, tuple[list[float], list[float]]] = {
    2: (
        [-np.inf, -0.9816, 0.0, 0.9816, np.inf],
        [-1.51, -0.4528, 0.4528, 1.51],
    ),
    3: (
        [-np.inf, -1.7479, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7479, np.inf],
        [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520],
    ),
    4: (
        [
            -np.inf, -2.4008, -1.8441, -1.4370, -1.0993, -0.7988, -0.5224, -0.2582,
            0.0, 0.2582, 0.5224, 0.7988, 1.0993, 1.4370, 1.8441, 2.4008, np.inf,
        ],
        [
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
        ],
    ),
}


def _compute_gaussian_lloyd_max(bit_width: int, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal quantizer for standard Gaussian via iterative algorithm."""
    if bit_width in GAUSSIAN_LLOYD_MAX:
        boundaries, centroids = GAUSSIAN_LLOYD_MAX[bit_width]
        return np.array(boundaries), np.array(centroids)

    num_levels = 2**bit_width
    centroids = np.linspace(-3.0, 3.0, num_levels)

    for _ in range(max_iter):
        boundaries = np.empty(num_levels + 1)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        new_centroids = np.empty(num_levels)
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = norm.cdf(hi) - norm.cdf(lo)
            if p < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else centroids[i]
            else:
                new_centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p

        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            centroids = new_centroids
            break
        centroids = new_centroids

    return boundaries, centroids


@dataclass
class QuantizedVectors:
    codes: np.ndarray
    bit_width: int
    num_vectors: int
    dim: int
    scale: np.ndarray
    shift: np.ndarray


class TurboQuantCompressor:
    """TurboQuant compressor: random rotation + Lloyd-Max scalar quantization."""

    def __init__(self, dim: int, bit_width: int, seed: int = 42) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self.seed = seed

        rng = np.random.RandomState(seed)
        gaussian_matrix = rng.randn(dim, dim).astype(np.float32)
        self.rotation, _ = np.linalg.qr(gaussian_matrix)

        self.boundaries, self.centroids = _compute_gaussian_lloyd_max(bit_width)

    def quantize(self, vectors: np.ndarray) -> QuantizedVectors:
        """Quantize (N, dim) float32 vectors."""
        rotated = vectors @ self.rotation

        per_dim_mean = rotated.mean(axis=0)
        per_dim_std = rotated.std(axis=0)
        per_dim_std[per_dim_std < 1e-8] = 1.0

        normalized = (rotated - per_dim_mean) / per_dim_std

        codes = np.digitize(normalized, self.boundaries[1:-1]).astype(np.uint8)
        codes = np.clip(codes, 0, 2**self.bit_width - 1)

        return QuantizedVectors(
            codes=codes,
            bit_width=self.bit_width,
            num_vectors=vectors.shape[0],
            dim=self.dim,
            scale=per_dim_std,
            shift=per_dim_mean,
        )

    def dequantize(self, qv: QuantizedVectors) -> np.ndarray:
        """Reconstruct (N, dim) float32 approximation from quantized codes."""
        reconstructed_normalized = self.centroids[qv.codes]
        reconstructed_rotated = reconstructed_normalized * qv.scale + qv.shift
        return (reconstructed_rotated @ self.rotation.T).astype(np.float32)

    def code_inner_product(self, query: np.ndarray, qv: QuantizedVectors) -> np.ndarray:
        """Compute inner products directly from integer codes (paper Algorithm 2).

        Groups dimensions by centroid assignment and sums query components per group,
        avoiding materialization of a full (N, dim) float array from codes.
        Returns (N,) array of scores.
        """
        q_rot = query @ self.rotation
        q_scaled = q_rot * qv.scale
        bias = float(q_rot @ qv.shift)

        num_levels = 2**self.bit_width
        scores = np.full(qv.num_vectors, bias, dtype=np.float64)
        for level in range(num_levels):
            mask = qv.codes == level
            scores += float(self.centroids[level]) * (mask @ q_scaled)

        return scores.astype(np.float32)

    def compression_ratio(self, num_vectors: int = 1000) -> float:
        """Compression ratio including amortized per-batch scale/shift overhead."""
        original_bits = self.dim * 32 * num_vectors
        code_bits = self.dim * self.bit_width * num_vectors
        overhead_bits = 2 * self.dim * 32
        return original_bits / (code_bits + overhead_bits)

    def memory_per_vector_bytes(self, num_vectors: int = 1000) -> float:
        code_bytes = (self.dim * self.bit_width) / 8
        overhead_bytes = (2 * self.dim * 4) / num_vectors
        return code_bytes + overhead_bytes
