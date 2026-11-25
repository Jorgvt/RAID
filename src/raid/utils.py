"""
raid.utils
----------
Hardware environment setup, CUDA shared library loaders, and path helpers.
"""

import os
import sys
import re
import ctypes
from pathlib import Path
from typing import List, Optional, Union
import torch


def setup_cuda_env():
    """
    Ensures NVIDIA PyTorch CUDA runtime shared libraries bundled inside
    the virtual environment are discovered and loaded into the process.
    """
    venv_site = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'nvidia'
    if not venv_site.exists():
        return
        
    lib_paths = [str(p) for p in venv_site.glob('*/lib') if p.is_dir()]
    if lib_paths:
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + (':' + current_ld if current_ld else '')
        for lib_dir in lib_paths:
            for so in Path(lib_dir).glob('*.so*'):
                try:
                    ctypes.CDLL(str(so))
                except Exception:
                    pass


def get_device(requested_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolves compute device. Defaults to CUDA GPU 0 if available, else CPU.
    """
    if requested_device is not None:
        if isinstance(requested_device, torch.device):
            return requested_device
        if 'cuda' in str(requested_device).lower() and not torch.cuda.is_available():
            setup_cuda_env()
        return torch.device(requested_device if (torch.cuda.is_available() or 'cuda' not in str(requested_device)) else 'cpu')
        
    setup_cuda_env()
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def find_tid2013_reference_images(ref_dir: Union[str, Path]) -> List[Path]:
    """
    Finds and sorts all reference images (I01..I25) in TID2013 reference_images directory.
    """
    ref_dir = Path(ref_dir)
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference images directory not found: {ref_dir}")
        
    ref_files = sorted(list(ref_dir.glob('*.BMP')) + list(ref_dir.glob('*.bmp')))
    unique_refs = list({p.stem.upper(): p for p in ref_files}.values())
    
    def extract_index(path: Path) -> int:
        match = re.search(r'\d+', path.stem)
        return int(match.group()) if match else 999
        
    return sorted(unique_refs, key=extract_index)


def resolve_image_path(base_dir: Union[str, Path], filename_or_stem: str) -> Path:
    """
    Resolves an image file inside base_dir taking into account case sensitivity (.bmp/.BMP).
    """
    base_dir = Path(base_dir)
    stem = Path(filename_or_stem).stem
    candidates = [
        base_dir / f"{stem.upper()}.BMP",
        base_dir / f"{stem.lower()}.bmp",
        base_dir / f"{stem}.bmp",
        base_dir / f"{stem}.BMP",
        base_dir / filename_or_stem
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Image '{filename_or_stem}' not found in '{base_dir}'.")
