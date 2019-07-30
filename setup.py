import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kaldi10-feats",
    version="0.0.1",
    author="Daniel Povey",
    author_email="dpovey@gmail.com",
    description="Kaldi10 speech features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danpovey/kaldi10-feats",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
