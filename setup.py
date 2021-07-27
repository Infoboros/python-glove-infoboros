import os
import subprocess

from setuptools import setup
from setuptools.command.install import install


class CustomInstall(install):
    def run(self):
        CC = 'gcc'
        CFLAGS = '-lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic'
        BUILDDIR = 'glove/build'
        SRCDIR = 'glove/src'

        os.makedirs(BUILDDIR, exist_ok=True)
        subprocess.check_call(
            [CC, '-c', SRCDIR+'/common.c', '-o', BUILDDIR+'/common.o'] + CFLAGS.split())
        subprocess.check_call(
            [CC, '-c', SRCDIR+'/vocab_count.c', '-o', BUILDDIR+'/vocab_count.o'] + CFLAGS.split())
        subprocess.check_call(
            [CC, '-c', SRCDIR+'/cooccur.c', '-o', BUILDDIR+'/cooccur.o'] + CFLAGS.split())
        subprocess.check_call(
            [CC, '-c', SRCDIR+'/shuffle.c', '-o', BUILDDIR+'/shuffle.o'] + CFLAGS.split())
        subprocess.check_call(
            [CC, '-c', SRCDIR+'/glove.c', '-o', BUILDDIR+'/glove.o'] + CFLAGS.split())

        subprocess.check_call(
            [CC, BUILDDIR+'/vocab_count.o', BUILDDIR+'/common.o', '-o', BUILDDIR+'/vocab_count'] + CFLAGS.split())
        subprocess.check_call(
            [CC, BUILDDIR+'/cooccur.o', BUILDDIR+'/common.o', '-o', BUILDDIR+'/cooccur'] + CFLAGS.split())
        subprocess.check_call(
            [CC, BUILDDIR+'/shuffle.o', BUILDDIR+'/common.o', '-o', BUILDDIR+'/shuffle'] + CFLAGS.split())
        subprocess.check_call(
            [CC, BUILDDIR+'/glove.o', BUILDDIR+'/common.o', '-o', BUILDDIR+'/glove'] + CFLAGS.split())
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
