import os

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as opened:
        return opened.read()


def main():
    setup(
        name='kerasfm',
        version='1.0.0',
        author="OrangeLe",
        author_email="raku.lxm@gmail.com",
        url='https://github.com/orangelle/kerasfm',
        description=('TensforFlow2.0 interpretation of arbitrary order '
                     'Factorization Machine'),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        license='MIT',
        install_requires=[
            'scikit-learn',
            'numpy',
            'tqdm'
        ],
        packages=find_packages()
    )


if __name__ == "__main__":
    main()
