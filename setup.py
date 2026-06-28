"""
Minimal setup.py for editable install in Google Colab.
Usage: pip install -e . (from the repo root)
"""
from setuptools import setup, find_packages

setup(
    name="hsi-lithology-classification",
    version="1.0.0",
    description=(
        "Baseline CNN showcase "
        "for Hyperspectral Image Classification of Geological Drill Cores"
    ),
    author="Suvarup Ghosh",
    author_email="suvarupghosh7916@gmail.com",
    url="https://github.com/suva1998/hsi-lithology-showcase",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "h5py>=3.8",
        "spectral>=0.23",
        "openpyxl>=3.1",
        "pandas>=2.0",
        "tqdm>=4.65",
    ],
)
