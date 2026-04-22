from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))


def _read_reqs(relpath):
    fullpath = path.join(path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")

setup(
    name="dexar_torch",
    version="1.0",
    description="DEX-AR: Explainability for Autoregressive Vision-Language Models",
    url="https://github.com/WalBouss/DEX-AR",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Explainability, Vision-Language Models, LLaVA, Heatmaps",
    packages=find_packages(exclude=["assets*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.8",
)
