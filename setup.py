from setuptools import setup, find_packages

setup(
    name="convrot-torchao",
    version="0.1.0",
    description="ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers",
    author="",
    url="https://github.com/YOUR_USERNAME/convrot-torchao",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.4.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "torchao>=0.8.0"],
        "flux": ["diffusers>=0.30.0", "transformers>=4.40.0", "accelerate>=0.30.0"],
        "eval": ["torchmetrics>=1.0", "lpips>=0.1"],
    },
)
