"""
Backward-compatibility setup script for renorm-native.
Enables legacy pip installations and editable developmental links.
"""

from setuptools import setup, find_packages

setup(
    name="renorm-native",
    version="1.0.0",
    packages=find_packages(include=["gateway", "layers", "scheduler", "loopguard"]),
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "triton": ["triton>=2.0.0"],
        "telemetry": ["prometheus-client>=0.16.0"],
    },
    python_requires=">=3.9",
)