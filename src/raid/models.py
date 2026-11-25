"""
raid.models
-----------
Calibrated perceptual metric wrappers, metric registry, and direct JOD predictors.
Supports standard metrics (RMSE, SSIM, LPIPS, DISTS) and all full-reference
metrics in the PIQ library (PSNR, MS-SSIM, IW-SSIM, VIF, FSIM, SRSIM, GMSD,
MS-GMSD, VSI, MDSI, HaarPSI, DSS, PieAPP, ContentLoss, StyleLoss).
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional, Callable, Sequence, List, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .psychometrics import logistic_4p, inverse_logistic_4p
from .utils import get_device

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------------
# Helper to register metrics
# ----------------------------------------------------------------------

def register_metric(name: str, compute_fn: Optional[Callable] = None,
                    formula: str = "", family: str = "custom",
                    piq_class: Optional[str] = None):
    """
    Decorator or function to register a new metric distance function:
    `compute_fn(img_ref, img_dist, device=None) -> float`.
    """
    def decorator(fn: Callable):
        METRIC_REGISTRY[name.upper()] = {
            'compute_fn': fn,
            'formula': formula or f"{name.upper()} distance",
            'family': family,
            'piq_class': piq_class
        }
        return fn

    if compute_fn is not None:
        return decorator(compute_fn)
    return decorator


def find_default_calibration_file() -> Path:
    """Finds calibration_parameters.json in current directory, parent dirs, or package data."""
    cwd_path = Path.cwd() / "calibration_parameters.json"
    if cwd_path.exists():
        return cwd_path
    repo_path = Path(__file__).resolve().parent.parent.parent / "calibration_parameters.json"
    if repo_path.exists():
        return repo_path
    return cwd_path


DEFAULT_PARAMS_PATH = find_default_calibration_file()


# ----------------------------------------------------------------------
# Built-in Baseline Distance Implementations
# ----------------------------------------------------------------------

@register_metric("RMSE", formula="Pixel RMSE (0-255 scale)", family="pixel")
def _compute_rmse(img_ref: Union[np.ndarray, Image.Image, str, Path],
                  img_dist: Union[np.ndarray, Image.Image, str, Path],
                  device: Optional[Any] = None) -> float:
    if isinstance(img_ref, (str, Path)):
        img_ref = Image.open(img_ref).convert("RGB")
    elif isinstance(img_ref, np.ndarray):
        img_ref = Image.fromarray(img_ref).convert("RGB")

    if isinstance(img_dist, (str, Path)):
        img_dist = Image.open(img_dist).convert("RGB")
    elif isinstance(img_dist, np.ndarray):
        img_dist = Image.fromarray(img_dist).convert("RGB")

    arr_ref = np.array(img_ref, dtype=float)
    arr_dist = np.array(img_dist, dtype=float)
    return float(np.sqrt(np.mean((arr_ref - arr_dist)**2)))


@register_metric("PSNR", formula="1 / max(PSNR, 1e-3)", family="pixel", piq_class="psnr")
def _compute_psnr_dist(img_ref: Any, img_dist: Any, device: Optional[Any] = None) -> float:
    import piq
    t_ref, t_dist = _to_tensor_pair(img_ref, img_dist, device=device)
    val = piq.psnr(t_ref, t_dist, data_range=1.0, reduction='none')
    return float(1.0 / torch.clamp(val, min=1e-3).item())


# ----------------------------------------------------------------------
# PIQ Metrics Registry Definitions
# ----------------------------------------------------------------------

PIQ_MODULE_CONFIGS: Dict[str, Dict[str, Any]] = {
    'SSIM': {'class': 'SSIMLoss', 'formula': '1 - SSIM', 'family': 'structural', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'MS_SSIM': {'class': 'MultiScaleSSIMLoss', 'formula': '1 - MS-SSIM', 'family': 'structural', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'IW_SSIM': {'class': 'InformationWeightedSSIMLoss', 'formula': '1 - IW-SSIM', 'family': 'structural', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'VIF': {'class': 'VIFLoss', 'formula': '1 - VIF_p', 'family': 'information', 'kwargs': {'reduction': 'none'}},
    'FSIM': {'class': 'FSIMLoss', 'formula': '1 - FSIM', 'family': 'gradient_saliency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'SRSIM': {'class': 'SRSIMLoss', 'formula': '1 - SRSIM', 'family': 'gradient_saliency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'GMSD': {'class': 'GMSDLoss', 'formula': 'GMSD deviation', 'family': 'gradient_saliency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'MS_GMSD': {'class': 'MultiScaleGMSDLoss', 'formula': 'MS-GMSD deviation', 'family': 'gradient_saliency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'VSI': {'class': 'VSILoss', 'formula': '1 - VSI', 'family': 'gradient_saliency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'MDSI': {'class': 'MDSILoss', 'formula': 'MDSI deviation', 'family': 'gradient_saliency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'HAARPSI': {'class': 'HaarPSILoss', 'formula': '1 - HaarPSI', 'family': 'wavelet', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'DSS': {'class': 'DSSLoss', 'formula': '1 - DSS', 'family': 'frequency', 'kwargs': {'data_range': 1.0, 'reduction': 'none'}},
    'LPIPS': {'class': 'LPIPS', 'formula': 'LPIPS deep distance (AlexNet)', 'family': 'deep_learned', 'kwargs': {'reduction': 'none'}},
    'DISTS': {'class': 'DISTS', 'formula': 'DISTS structure & texture distance', 'family': 'deep_learned', 'kwargs': {'reduction': 'none'}},
    'PIEAPP': {'class': 'PieAPP', 'formula': 'PieAPP perceptual error', 'family': 'deep_learned', 'kwargs': {'reduction': 'none'}},
    'CONTENT_LOSS': {'class': 'ContentLoss', 'formula': 'VGG16 feature MSE', 'family': 'deep_learned', 'kwargs': {}},
    'STYLE_LOSS': {'class': 'StyleLoss', 'formula': 'VGG16 Gram matrix MSE', 'family': 'deep_learned', 'kwargs': {}},
}

for name, cfg in PIQ_MODULE_CONFIGS.items():
    if name not in METRIC_REGISTRY:
        METRIC_REGISTRY[name] = {
            'compute_fn': None,
            'formula': cfg['formula'],
            'family': cfg['family'],
            'piq_class': cfg['class'],
            'kwargs': cfg.get('kwargs', {})
        }

ALL_AVAILABLE_METRICS = list(METRIC_REGISTRY.keys())


def _to_tensor_pair(img_ref: Any, img_dist: Any, device: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to convert image inputs to normalized (1, 3, H, W) PyTorch tensors on device."""
    dev = get_device(device)
    to_tensor = transforms.ToTensor()

    if isinstance(img_ref, (str, Path)):
        img_ref = Image.open(img_ref).convert("RGB")
    elif isinstance(img_ref, np.ndarray):
        img_ref = Image.fromarray(img_ref.astype(np.uint8)).convert("RGB")

    if isinstance(img_dist, (str, Path)):
        img_dist = Image.open(img_dist).convert("RGB")
    elif isinstance(img_dist, np.ndarray):
        img_dist = Image.fromarray(img_dist.astype(np.uint8)).convert("RGB")

    t_ref = to_tensor(img_ref).unsqueeze(0).to(dev)
    t_dist = to_tensor(img_dist).unsqueeze(0).to(dev)
    return t_ref, t_dist


def instantiate_metric_model(metric_name: str, device: Optional[Any] = None) -> Optional[torch.nn.Module]:
    """Instantiates and returns a torch.nn.Module for a metric if it is PIQ-based."""
    m_key = metric_name.upper()
    if m_key not in METRIC_REGISTRY:
        return None

    cfg = METRIC_REGISTRY[m_key]
    piq_cls_name = cfg.get('piq_class')
    if not piq_cls_name:
        return None

    import piq
    if not hasattr(piq, piq_cls_name):
        return None

    cls_obj = getattr(piq, piq_cls_name)
    if isinstance(cls_obj, type) and issubclass(cls_obj, torch.nn.Module):
        kwargs = cfg.get('kwargs', {})
        dev = get_device(device)
        model = cls_obj(**kwargs).to(dev).eval()
        return model
    return None


class CalibratedMetric:
    """
    Wrapper for a perceptual quality metric calibrated to human JOD scale.
    Supports built-in and custom registered metrics.
    """
    def __init__(self, metric_name: str, standard: str, params: Dict[str, Any],
                 custom_compute_fn: Optional[Callable] = None,
                 device: Optional[Union[str, Any]] = None):
        self.metric_name = metric_name.upper()
        self.standard = standard
        self.params = params
        self.b1 = float(params["b1"])
        self.b2 = float(params["b2"])
        self.b3 = float(params["b3"])
        self.b4 = float(params["b4"])
        self.jnd_1_0_d = params.get("jnd_1_0_d", None)
        self.jnd_1_0_raw = params.get("jnd_1_0_raw", None)
        self.device = get_device(device)
        self.custom_compute_fn = custom_compute_fn
        self._model = None

    @classmethod
    def load(cls, metric_name: str = "SSIM", standard: str = "gaussian_noise_baseline",
             device: Optional[Union[str, Any]] = None, params_path: Union[str, Path] = DEFAULT_PARAMS_PATH):
        params_path = Path(params_path)
        if not params_path.exists():
            params_path = find_default_calibration_file()
        if not params_path.exists():
            raise FileNotFoundError(f"Calibration parameter file not found: {params_path}")

        with open(params_path, "r") as f:
            all_params = json.load(f)

        metric_key = metric_name.upper()
        if metric_key not in all_params:
            raise KeyError(f"Metric '{metric_name}' not found in parameters. Available: {list(all_params.keys())}")
        if standard not in all_params[metric_key]:
            raise KeyError(f"Standard '{standard}' not found for metric '{metric_name}'. Available: {list(all_params[metric_key].keys())}")

        return cls(metric_key, standard, all_params[metric_key][standard], device=device)

    def _init_model_if_needed(self):
        if self._model is not None:
            return
        self._model = instantiate_metric_model(self.metric_name, device=self.device)

    def distance_to_jod(self, d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Maps raw metric distance d to predicted human JOD."""
        return logistic_4p(d, self.b1, self.b2, self.b3, self.b4)

    def jod_to_distance(self, jod: float) -> float:
        """Calculates raw distance d required to achieve a target JOD value."""
        return inverse_logistic_4p(jod, self.b1, self.b2, self.b3, self.b4)

    def compute_raw_distance(self, img_ref: Union[np.ndarray, Image.Image, str, Path],
                             img_dist: Union[np.ndarray, Image.Image, str, Path]) -> float:
        """Computes raw metric distance d between reference and distorted image."""
        if self.custom_compute_fn is not None:
            return float(self.custom_compute_fn(img_ref, img_dist, device=self.device))

        if self.metric_name in METRIC_REGISTRY and METRIC_REGISTRY[self.metric_name]['compute_fn'] is not None:
            return float(METRIC_REGISTRY[self.metric_name]['compute_fn'](img_ref, img_dist, device=self.device))

        self._init_model_if_needed()
        if self._model is not None:
            t_ref, t_dist = _to_tensor_pair(img_ref, img_dist, device=self.device)
            with torch.no_grad():
                res = self._model(t_ref, t_dist)
                if isinstance(res, torch.Tensor):
                    res = float(res.squeeze().item())
            return float(res)

        raise NotImplementedError(f"Metric '{self.metric_name}' is not registered and has no compute function.")

    def compute_batch_distance(self, ref_batch: torch.Tensor, dist_batch: torch.Tensor) -> np.ndarray:
        """Computes batched distance for (B, C, H, W) tensors on device."""
        self._init_model_if_needed()
        if self._model is not None:
            with torch.no_grad():
                res = self._model(ref_batch, dist_batch)
                if isinstance(res, torch.Tensor):
                    arr = res.squeeze().detach().cpu().numpy()
                    if arr.ndim == 0:
                        arr = np.array([float(arr)])
                    return arr
        # Fallback for CPU / custom functions
        distances = []
        for i in range(ref_batch.shape[0]):
            r_pil = transforms.ToPILImage()(ref_batch[i].cpu())
            d_pil = transforms.ToPILImage()(dist_batch[i].cpu())
            distances.append(self.compute_raw_distance(r_pil, d_pil))
        return np.array(distances, dtype=float)

    def predict_jod(self, img_ref: Union[np.ndarray, Image.Image, str, Path],
                    img_dist: Union[np.ndarray, Image.Image, str, Path]) -> float:
        """Computes metric distance and converts directly to predicted JOD units."""
        d = self.compute_raw_distance(img_ref, img_dist)
        return float(self.distance_to_jod(d))


def save_calibration_parameters(params_dict: Dict[str, Any], filepath: Union[str, Path] = DEFAULT_PARAMS_PATH):
    """Saves or updates calibration parameters in local JSON file."""
    filepath = Path(filepath)
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    for metric, standards in params_dict.items():
        if metric not in existing:
            existing[metric] = {}
        for std_name, std_data in standards.items():
            existing[metric][std_name] = std_data

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Calibration parameters saved to: {filepath}")

