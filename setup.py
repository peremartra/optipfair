from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optipfair",
    version="0.1.2",
    author="Pere Martra",
    author_email="peremartra@uadla.com",
    description="A library for structured pruning & Bias visualization of large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peremartra/optipfair",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.25.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.900",
        ],
        "eval": [
            "datasets>=2.0.0",
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "optipfair=optipfair.cli:cli",
        ],
    },
)