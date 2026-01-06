"""Setup script for Image-to-Text Generator."""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="image-to-text-generator",
    version="2.0.0",
    author="Tharun Ponnam",
    author_email="tharunponnam007@gmail.com",
    description="Automatic image captioning using BLIP vision-language model",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/tharun-ship-it/image-to-text-generator",
    packages=find_packages(exclude=["tests", "notebooks", "assets"]),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "image-caption=src.inference.generator:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=["image-captioning", "blip", "vision-language", "deep-learning", "streamlit"],
)
