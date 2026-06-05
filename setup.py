"""Minimal setup.py for compatibility with older pip versions.
All config is in pyproject.toml; this file exists only as a fallback
for servers where pip < 21.3 doesn't fully support PEP 621.
"""
from setuptools import setup, find_packages

setup(
    name="herpb",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    ],
)
