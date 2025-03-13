from setuptools import setup, find_packages

setup(
    name="ner_ancient_swedish",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "datasets>=3.3.2",
        "nervaluate>=0.2.0",
        "seqeval>=1.2.2",
        "pandas>=2.2.3",
        "numpy>=2.2.3",
        "scikit-learn>=1.6.1",
    ],
    author="F.V.",
    author_email="fvagnoni.ieu2021@student.ie.edu",
    description="NER for Ancient Swedish",
    keywords="NER, Swedish, EuroBERT, Natural Language Processing",
    url="https://github.com/yourusername/NER_Ancient_Swedish",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 