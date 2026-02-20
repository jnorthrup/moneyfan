"""
Convergence detection over model signal histories.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

import numpy as np


def convergence_from_snapshot(
    signals: np.ndarray,
    confidences: np.ndarray,
    *,
    min_agree: int = 2,
    min_confidence_sum: float = 0.25,
) -> float:
    """
    Compute a convergence score in [0, 1] for one signal snapshot.
    """
    sig = np.asarray(signals, dtype=np.float32).reshape(-1)
    conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if sig.size == 0 or conf.size == 0 or sig.size != conf.size:
        return 0.0

    pos = sig > 0
    neg = sig < 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        return 0.0

    dominant_count = max(n_pos, n_neg)
    if dominant_count < min_agree:
        return 0.0

    pos_conf = float(np.sum(conf[pos])) if n_pos > 0 else 0.0
    neg_conf = float(np.sum(conf[neg])) if n_neg > 0 else 0.0
    dominant_conf = max(pos_conf, neg_conf)
    if dominant_conf < min_confidence_sum:
        return 0.0

    agreement_ratio = dominant_count / float(n_nonzero)
    total_conf = pos_conf + neg_conf
    if total_conf <= 1e-8:
        return 0.0
    confidence_lopsided = abs(pos_conf - neg_conf) / total_conf

    score = agreement_ratio * confidence_lopsided
    return float(np.clip(score, 0.0, 1.0))


def rolling_convergence(
    signals_hist: np.ndarray,
    confidences_hist: np.ndarray,
    *,
    window: int = 32,
    min_agree: int = 2,
    min_confidence_sum: float = 0.25,
) -> np.ndarray:
    """
    Rolling convergence scores for historical signal arrays.

    Args:
        signals_hist: [T, M]
        confidences_hist: [T, M]
    """
    sig = np.asarray(signals_hist, dtype=np.float32)
    conf = np.asarray(confidences_hist, dtype=np.float32)
    if sig.ndim != 2 or conf.ndim != 2 or sig.shape != conf.shape:
        raise ValueError("signals_hist and confidences_hist must both be [T, M] with same shape")

    t_len = sig.shape[0]
    out = np.zeros((t_len,), dtype=np.float32)
    if t_len == 0:
        return out

    w = max(1, int(window))
    for t in range(t_len):
        start = max(0, t - w + 1)
        s_window = sig[start : t + 1].reshape(-1)
        c_window = conf[start : t + 1].reshape(-1)
        out[t] = convergence_from_snapshot(
            s_window,
            c_window,
            min_agree=min_agree,
            min_confidence_sum=min_confidence_sum,
        )
    return out


@dataclass
class ModelSignalConvergenceTracker:
    """
    Tracks per-instrument convergence over rolling model-signal history.
    """

    n_instruments: int
    n_models: int
    window: int = 32
    min_agree: int = 2
    min_confidence_sum: float = 0.25

    def __post_init__(self):
        self.window = max(1, int(self.window))
        self._signals: List[Deque[np.ndarray]] = [
            deque(maxlen=self.window) for _ in range(self.n_instruments)
        ]
        self._confidences: List[Deque[np.ndarray]] = [
            deque(maxlen=self.window) for _ in range(self.n_instruments)
        ]

    def update(self, snapshot: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update tracker with one timestep snapshot.

        Args:
            snapshot: [N, M, F] where [:,:,0]=signal and [:,:,1]=confidence
        """
        arr = np.asarray(snapshot, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("snapshot must be [N, M, F]")
        if arr.shape[0] != self.n_instruments:
            raise ValueError(f"snapshot instruments {arr.shape[0]} != tracker {self.n_instruments}")
        if arr.shape[1] != self.n_models:
            raise ValueError(f"snapshot models {arr.shape[1]} != tracker {self.n_models}")
        if arr.shape[2] < 2:
            raise ValueError("snapshot feature dimension must include signal and confidence at indices 0,1")

        sig = arr[:, :, 0]
        conf = np.clip(arr[:, :, 1], 0.0, 1.0)

        convergence = np.zeros((self.n_instruments,), dtype=np.float32)
        agreement_ratio = np.zeros((self.n_instruments,), dtype=np.float32)
        confidence_support = np.zeros((self.n_instruments,), dtype=np.float32)

        for i in range(self.n_instruments):
            self._signals[i].append(sig[i].copy())
            self._confidences[i].append(conf[i].copy())

            sig_hist = np.stack(self._signals[i], axis=0)
            conf_hist = np.stack(self._confidences[i], axis=0)
            convergence[i] = rolling_convergence(
                sig_hist,
                conf_hist,
                window=self.window,
                min_agree=self.min_agree,
                min_confidence_sum=self.min_confidence_sum,
            )[-1]

            s_now = sig[i]
            c_now = conf[i]
            pos = s_now > 0
            neg = s_now < 0
            n_pos = int(np.sum(pos))
            n_neg = int(np.sum(neg))
            n_nonzero = max(1, n_pos + n_neg)
            agreement_ratio[i] = float(max(n_pos, n_neg) / n_nonzero)
            confidence_support[i] = float(max(np.sum(c_now[pos]), np.sum(c_now[neg])))

        return {
            "convergence": convergence,
            "agreement_ratio": agreement_ratio,
            "confidence_support": confidence_support,
        }

