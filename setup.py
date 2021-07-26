import subprocess

from setuptools import setup
from setuptools.command.install import install


class CustomInstall(install):
    def run(self):
        subprocess.check_call(['make'], cwd='glove', shell=True)
        super().run()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="python-glove",
    version="0.0.1",
    description="Python implementation of https://nlp.stanford.edu/projects/glove/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iconclub/python-glove",
    author="Hieu Nguyen",
    author_email="hieunguyen1053@outlook.com",
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["glove"],
    include_package_data=True,
    install_requires=["numpy==1.21.1"],
    entry_points={},
    cmdclass={'install': CustomInstall}
)
