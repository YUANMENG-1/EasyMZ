from setuptools import setup

setup(
    name="pyMS_Few",
    version="0.2.0",
    packages=["pyMS_Few"],
    include_package_data=True,
    package_data={
        "pyMS_Few": [
            "binaries/*",
            "scripts/*.py",
            "main.py",
            "__init__.py"
        ]
    },
    install_requires=[
        "pyopenms>=3.4.0",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "matplotlib>=3.9.4"
    ],
    entry_points={
        "console_scripts": [
            "pyMS_Few = pyMS_Few.main:main"
        ]
    },
    author="YuanMY",
    author_email="your.email@example.com",
    description="A pipeline for processing mzML files on macOS Intel x86_64",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: MacOS :: MacOS X",
        "Environment :: MacOS X",
    ],
    python_requires=">=3.6",
)