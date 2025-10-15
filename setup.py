"""
WarpFactory Python Package Setup
A numerical toolkit for analyzing warp drive spacetimes using Einstein's General Relativity
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="warpfactory",
    version="1.0.0",
    author="Christopher Helmerich, Jared Fuchs, and Contributors",
    author_email="",
    description="Numerical toolkit for analyzing warp drive spacetimes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NerdsWithAttitudes/WarpFactory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "gpu": ["cupy>=10.0.0"],
        "viz": ["pyvista>=0.38.0", "vtk>=9.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    package_data={
        "warpfactory": ["py.typed"],
    },
)
