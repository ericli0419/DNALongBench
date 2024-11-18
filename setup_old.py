from setuptools import setup, find_packages

setup(
    name="dnalongbench",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for DNA long benchmark analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wenduocheng/DNALongBench.git",  # Replace with your repository URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "selene-sdk==0.5.3",
        "torchmetrics",
        "kipoiseq==0.5.2",
        "biopython",
        "pandas==2.1.4",
        "cython",
        "scipy==1.12.0",
        "matplotlib==3.8",
        "pyBigWig",
        "tensorflow==2.12.0",
        "typing_extensions==4.12.2",
        "torch==2.1.0+cu118",
        "torchvision==0.16.0+cu118",
        "torchaudio==2.1.0+cu118",
        "torchtext==0.16.0",
        "numpy==1.26.4",
        "natsort",
        "pytabix",
        "tqdm",
    ]
)
