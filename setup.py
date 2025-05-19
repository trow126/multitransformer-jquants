from setuptools import setup, find_packages

setup(
    name="multitransformer-jquants",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.4.0",
        "numpy>=1.19.5",
        "pandas>=1.2.0",
        "matplotlib>=3.3.4",
        "seaborn>=0.11.1",
        "arch>=4.19",
        "jquantsapi>=0.1.0",
        "pyyaml>=5.4.1",
        "tqdm>=4.61.0",
        "scikit-learn>=0.24.2",
        "python-dotenv>=0.19.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="MultiTransformerモデルをJ-Quantsデータに適用するプロジェクト",
    keywords="transformer, finance, stock, volatility, prediction, jquants",
    url="https://github.com/yourusername/multitransformer-jquants",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)