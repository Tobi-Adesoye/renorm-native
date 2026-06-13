from setuptools import setup, find_packages

setup(
    name="renorm-native",
    version="1.0.0",
    # This instructs setuptools to treat the 'src' directory as the root folder
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
    ],
)