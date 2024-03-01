import os
from setuptools import setup, find_packages


# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


"""
BM: Added these requirements (up until the next #) to assist Github Actions. They need to be the specified versions since:
    * The tests break if stable baselines is upgraded beyond 1.5.0
    * Stable baselines 1.5.0 depends on gym==0.21
    * gym==0.21 depends on setuptools<=65.5.0 and pip<=21 - https://github.com/openai/gym/issues/3176#issuecomment-1560026649
"""
requirements = [
    "setuptools==65.5.0",
    "pip==21",
    #
    "wheel>=0.38.0",
    "stable-baselines3==1.5.0",
    "tabulate>=0.8.0",
    "tensorboard>=2.9.0",
    "networkx>=2.8.4"
]

setup(
    name="NFVdeep",
    description="Deep Reinforcement Learning for Online Orchestration of Service Function Chains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CN-UPB/NFVdeep",
    packages=find_packages(),
    python_requires=">=3.8", #BM: I changed this to function with higher versions of python
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
