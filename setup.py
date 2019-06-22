from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

from setuptools import find_packages, setup


loader = SourceFileLoader("tmexp", "./tmexp/__init__.py")
tmexp = ModuleType(loader.name)
loader.exec_module(tmexp)


setup(
    name="tmexp",
    version=tmexp.__version__,  # type: ignore
    description="Topic Modeling on Source Code Experiments.",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="source{d}'s Research Machine Learning Team",
    author_email="research-machine-learning@sourced.tech",
    python_requires=">=3.6.0",
    url="https://github.com/src-d/tm-experiments",
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["tmexp=tmexp.__main__:main"]},
    install_requires=["nltk==3.4.3", "pymysql==0.9.3", "bblfsh==3.0.4", "tqdm==4.32.2"],
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
