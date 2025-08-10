from setuptools import setup, find_packages

setup(
    name="sac-trading-agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)