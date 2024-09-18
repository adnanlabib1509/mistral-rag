from setuptools import setup, find_packages

setup(
    name="mistral_rag",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "numpy",
        "PyPDF2",
        "pdf2image",
        "pytesseract",
    ],
    author="Adnan",
    author_email="adnanlabib1509@gmail.com",
    description="A Retrieval-Augmented Generation system using Mistral AI's language model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adnanlabib1509/mistral-rag",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)