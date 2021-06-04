import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "ngocbienml",
    version="1.1.5",
    author="Nguyen Ngoc Bien",
    author_email="ngocbien.nguyen.vn@gmail.com",
    description="An ecosystem for machine learning project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/ngocbien/ngocbienml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.*',
    license='MIT',
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-optimize'
    ]
)