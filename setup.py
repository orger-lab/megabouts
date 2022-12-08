import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="megabouts",
    version="0.0.1",
    author="Adrien Jouary & Alexandre Laborde",
    author_email="adrien.jouary@research.fchampalimaud.org",
    description="Quantification of Zebrafish larva behavior",
    url="https://github.com/orger-lab/megabouts",
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy",
        "pandas",
        "pybaselines",
        "roipoly",
        "scikit_image",
        "scikit_learn",
        "scipy",
        "setuptools",
        "sporco",
    ],
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.npz","*.npy","*.pickle"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'pysoft = pysoft.main:main',
        ],
    },
    python_requires='>=3.7',
)