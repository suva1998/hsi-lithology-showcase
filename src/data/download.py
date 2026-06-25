"""
Dataset loading utilities for raw ENVI format geological HSI benchmarks.

Specifically handles the HZDR RODARE Upscaling Mineralogy Benchmark
(Thiele et al., 2026), parsing raw FENIX (.dat/.hdr) hyperspectral cubes
and corresponding MLA fractional abundance maps into discrete lithology labels.
"""

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import spectral.io.envi as envi
import pandas as pd


def get_hyc_directories(data_dir: str, filter_str: str = "Stonepark") -> List[str]:
    """Find all .hyc directories containing drill core samples."""
    hyc_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if root.endswith('.hyc') and "FENIX.dat" in files and "MLA_HSI.dat" in files:
            if filter_str and filter_str.lower() not in root.lower():
                continue
            hyc_dirs.append(root)
    return hyc_dirs


def load_envi_dataset(
    data_dir: str, 
    mapping_excel: str = None,
    max_cores: int = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load raw ENVI FENIX hyperspectral data and MLA ground truth.

    Args:
        data_dir: Root directory to search for .hyc sample folders.
        mapping_excel: Path to AbundanceMapping.xlsx. If None, tries to find it.

    Returns:
        Tuple of (stacked_data, stacked_labels, class_names)
            stacked_data: shape (Total_H, W, Bands)
            stacked_labels: shape (Total_H, W) discrete class indices
            class_names: list of string class names
    """
    data_dir = Path(data_dir)
    
    if mapping_excel is None:
        # Search for the excel file in the root data dir
        excel_matches = list(data_dir.rglob("AbundanceMapping.xlsx"))
        if not excel_matches:
            raise FileNotFoundError("Could not find AbundanceMapping.xlsx")
        mapping_excel = excel_matches[0]

    print(f"Loading mineral mapping from: {mapping_excel}")
    df = pd.read_excel(mapping_excel)
    
    # Extract unique macro groups (ignoring NaNs)
    macro_groups = df['Assigned mineral group'].dropna().unique().tolist()
    # Add a 'Background' class at index 0
    class_names = ["Background"] + macro_groups
    
    mineral_to_macro = {}
    for _, row in df.iterrows():
        min_name = str(row['Mapped Mineral (MLA)']).strip()
        macro_group = str(row['Assigned mineral group']).strip()
        if pd.isna(row['Mapped Mineral (MLA)']) or 'N.B.' in min_name:
            continue
        if pd.isna(row['Assigned mineral group']) or macro_group == 'nan':
            continue
        mineral_to_macro[min_name] = class_names.index(macro_group)

    hyc_dirs = get_hyc_directories(str(data_dir))
    if not hyc_dirs:
        raise ValueError(f"No valid .hyc directories found in {data_dir} containing FENIX.dat and MLA_HSI.dat")

    if max_cores is not None:
        hyc_dirs = hyc_dirs[:max_cores]

    import concurrent.futures

    def _process_single_core(hyc_path: str) -> Tuple[np.ndarray, np.ndarray]:
        fenix_hdr = os.path.join(hyc_path, "FENIX.hdr")
        fenix_dat = os.path.join(hyc_path, "FENIX.dat")
        img = envi.open(fenix_hdr, fenix_dat)
        hsi_cube = img.load().astype(np.float32)
        
        mla_hdr = os.path.join(hyc_path, "MLA_HSI.hdr")
        mla_dat = os.path.join(hyc_path, "MLA_HSI.dat")
        mla_img = envi.open(mla_hdr, mla_dat)
        mla_cube = mla_img.load()
        
        band_names = mla_img.metadata.get('band names', [])
        band_to_macro = np.zeros(mla_cube.shape[-1], dtype=np.int32)
        for i, b_name in enumerate(band_names):
            clean_name = b_name.strip()
            if clean_name in mineral_to_macro:
                band_to_macro[i] = mineral_to_macro[clean_name]
                
        dominant_band = np.argmax(mla_cube, axis=-1)
        max_abundance = np.max(mla_cube, axis=-1)
        discrete_labels = band_to_macro[dominant_band]
        discrete_labels[max_abundance == 0] = 0
        
        mask_hdr = os.path.join(hyc_path, "mask.hdr")
        mask_dat = os.path.join(hyc_path, "mask.dat")
        if os.path.exists(mask_hdr):
            mask_img = envi.open(mask_hdr, mask_dat)
            valid_mask = mask_img.load().squeeze()
            min_h = min(discrete_labels.shape[0], valid_mask.shape[0])
            min_w = min(discrete_labels.shape[1], valid_mask.shape[1])
            discrete_labels[:min_h, :min_w][valid_mask[:min_h, :min_w] == 0] = 0
            hsi_cube = hsi_cube[:min_h, :min_w, :]
            discrete_labels = discrete_labels[:min_h, :min_w]
            
        return np.nan_to_num(hsi_cube), discrete_labels

    print(f"Found {len(hyc_dirs)} drill core samples. Loading in parallel (this is much faster on Colab)...")
    all_data = []
    all_labels = []

    max_workers = min(32, (os.cpu_count() or 1) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_core, hyc) for hyc in hyc_dirs]
        for future in concurrent.futures.as_completed(futures):
            try:
                hsi, labels = future.result()
                all_data.append(hsi)
                all_labels.append(labels)
            except Exception as e:
                print(f"Error loading a core: {e}")

    # For simplicity in this benchmark pipeline, we stack them vertically.
    
    # Find the maximum width across all cores
    if not all_data:
        raise ValueError("No valid data loaded from .hyc directories.")
        
    max_width = max(d.shape[1] for d in all_data)
    padded_data = []
    padded_labels = []
    
    # Pad all cores to the same width so they can be vertically stacked
    for d, l in zip(all_data, all_labels):
        pad_width = max_width - d.shape[1]
        if pad_width > 0:
            d = np.pad(d, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            l = np.pad(l, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        padded_data.append(d)
        padded_labels.append(l)

    stacked_data = np.vstack(padded_data)
    stacked_labels = np.vstack(padded_labels)

    print(f"Loaded {len(all_data)} core segments:")
    print(f"   Total Data shape:  {stacked_data.shape}  (H × W × Bands)")
    print(f"   Total Label shape: {stacked_labels.shape}")
    print(f"   Classes mapped:    {len(class_names)-1} (excl. background)")
    
    return stacked_data, stacked_labels, class_names

# Backwards compatibility alias for the old pipeline
def download_dataset(name: str, data_dir: str = "data/raw") -> Tuple[np.ndarray, np.ndarray]:
    data, labels, _ = load_envi_dataset(data_dir)
    return data, labels
