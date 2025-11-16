"""Setup script for Fractal Glyph Tape."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="fractal-glyph-tape",
    version="0.1.0",
    author="Glyphd Labs",
    author_email="contact@glyphd.com",
    description="A fractal-addressable phrase memory for semantic compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/funwae/fractal-glyph-tape",
    project_urls={
        "Bug Tracker": "https://github.com/funwae/fractal-glyph-tape/issues",
        "Documentation": "https://github.com/funwae/fractal-glyph-tape/tree/main/docs",
        "Source Code": "https://github.com/funwae/fractal-glyph-tape",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fgt=fgt.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
