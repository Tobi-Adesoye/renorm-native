from setuptools import setup, find_packages

setup(
    name="renorm-native",
    version="1.0.0",
    description="Deep Functional Manifolds: Systemic Corrections for Identity Residual Explosions",
    author="Renorm Architecture Group",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)