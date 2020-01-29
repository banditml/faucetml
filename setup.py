import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faucetml",
    version="0.0.3",
    author="Edoardo Conti, Lionel Vital",
    author_email="edoardo.conti@gmail.com",
    description="Simple, high-speed batch data reader & preprocessor for ML applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/econti/FaucetML",
    download_url="https://github.com/econti/FaucetML/archive/v0.0.3.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>0.20.0",
        "feast>=0.4.4",
        "google-cloud-bigquery",
        "retry",
        "torch",
        "scipy",
    ],
)
