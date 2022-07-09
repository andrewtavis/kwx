try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os

from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name="kwx",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="1.0.1",
    author="Andrew Tavis McAllister",
    author_email="andrew.t.mcallister@gmail.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    description="BERT, LDA, and TFIDF based keyword extraction in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="new BSD",
    url="https://github.com/andrewtavis/kwx",
)

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    install_requires = []
else:
    install_requires = [
        "pytest-cov",
        "numpy",
        "xlrd",
        "pandas",
        "matplotlib",
        "seaborn",
        "stopwordsiso",
        "gensim",
        "pyldavis",
        "wordcloud",
        "nltk",
        "spacy",
        "emoji",
        "googletrans",
        "scikit-learn",
        "keras",
        "IPython",
        "sentence-transformers",
        "tqdm",
        "defusedxml",
    ]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)
