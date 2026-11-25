"""
raid.psychometrics
------------------
Standard ITU-R BT.500 / VQEG 4-parameter logistic mapping, optimization,
JND threshold extraction, and psychometric benchmark evaluation metrics.
"""

from typing import Dict, Any, Tuple, Optional, Sequence, Union
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau


def logistic_4p(d: Union[float, np.ndarray], b1: float, b2: float, b3: float, b4: float) -> Union[float, np.ndarray]:
    """
    ITU-R BT.500 / VQEG 4-Parameter Logistic Mapping function.
    Maps raw distance d monotonically to predicted human JOD:
        f(d) = b2 + (b1 - b2) / (1 + exp(-(d - b3) / |b4|))
    """
    d_arr = np.asarray(d, dtype=float)
    b4_safe = np.maximum(np.abs(b4), 1e-6)
    z = np.clip(-(d_arr - b3) / b4_safe, -50.0, 50.0)
    res = b2 + (b1 - b2) / (1.0 + np.exp(z))
    if np.ndim(d) == 0:
        return float(res)
    return res


def inverse_logistic_4p(jod: float, b1: float, b2: float, b3: float, b4: float) -> float:
    """
    Invert 4-parameter logistic mapping: target JOD -> raw distance d.
    Solves f(d) = jod for d.
    """
    b4_safe = max(abs(b4), 1e-6)
    ratio = (b1 - b2) / (jod - b2) - 1.0
    if ratio <= 0:
        return np.nan
    return float(b3 - b4_safe * np.log(ratio))


def fit_logistic_4p(x: Sequence[float], y: Sequence[float], p0: Optional[Sequence[float]] = None, maxfev: int = 35000) -> np.ndarray:
    """
    Fits parameters [b1, b2, b3, b4] using non-linear least squares (Levenberg-Marquardt).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = [float(np.max(y_arr)), float(np.min(y_arr)), float(np.median(x_arr)), float(max(np.std(x_arr), 0.1))]
    popt, _ = curve_fit(logistic_4p, x_arr, y_arr, p0=p0, maxfev=maxfev)
    return popt


def evaluate_predictions(y_true: Sequence[float], y_pred: Sequence[float], raw_d: Sequence[float]) -> Dict[str, float]:
    """
    Evaluates correlation and error metrics between true human JOD and model predictions:
    - SROCC: Spearman Rank Order Correlation Coefficient
    - KROCC: Kendall Rank Order Correlation Coefficient
    - PLCC: Pearson Linear Correlation Coefficient
    - RMSE_JOD: Root Mean Square Error in JOD units
    - MAE_JOD: Mean Absolute Error in JOD units
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    raw_d_arr = np.asarray(raw_d, dtype=float)

    srocc, _ = spearmanr(raw_d_arr, y_true_arr)
    krocc, _ = kendalltau(raw_d_arr, y_true_arr)
    plcc = pearsonr(y_pred_arr, y_true_arr)[0] if np.std(y_pred_arr) > 1e-7 else 0.0
    rmse = np.sqrt(np.mean((y_pred_arr - y_true_arr)**2))
    mae = np.mean(np.abs(y_pred_arr - y_true_arr))

    return {
        'SROCC': float(srocc),
        'KROCC': float(krocc),
        'PLCC': float(plcc),
        'RMSE_JOD': float(rmse),
        'MAE_JOD': float(mae)
    }


def compute_jnd_threshold(stimuli_levels: Sequence[float], jod_curve: Sequence[float], target_jod: float = 1.0) -> float:
    """
    Extracts the stimulus magnitude (e.g. translation shift in pixels, rotation in degrees)
    required to reach target_jod (default: 1.0 JOD = 1 JND threshold) via exact linear interpolation.
    """
    stimuli_arr = np.asarray(stimuli_levels, dtype=float)
    jod_arr = np.asarray(jod_curve, dtype=float)
    
    if len(stimuli_arr) == 0 or len(jod_arr) == 0:
        return np.nan

    if jod_arr[0] >= target_jod:
        # Linear approximation through origin (0, 0)
        return float(target_jod / jod_arr[0] * stimuli_arr[0])
    
    cross = np.where(jod_arr >= target_jod)[0]
    if len(cross) > 0:
        i = cross[0]
        x0, x1 = stimuli_arr[i - 1], stimuli_arr[i]
        y0, y1 = jod_arr[i - 1], jod_arr[i]
        if y1 == y0:
            return float(x0)
        return float(x0 + (target_jod - y0) * (x1 - x0) / (y1 - y0))
    return np.nan
