from setuptools import setup

VERSION = "0.0.12"
DESCRIPTION = "Very Basic pacjage to store results of ML models"
LONG_DESCRIPTION = """
Grid search results, are hard to exploit.
This package aims to store them in a more convenient way
"""

setup(
    name="skres",
    version=VERSION,
    author="Alex Gazagnes",
    author_email="<alex.gaz@email.com>",
    url="https://github.com/AlexandreGazagnes/skres",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # packages=find_packages(),
    packages=["skres"],
    install_requires=["pandas"],
    keywords=["python", "machine learning", "sklearn", "results"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
