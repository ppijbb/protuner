import setuptools

with open("Readme.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ProTuner",
    version="0.0.1",
    author="Example Author",
    author_email="kevin.jung@eturingbio.com",
    description="ProTuner Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/turingbio/protuner",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)