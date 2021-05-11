from setuptools import find_packages, setup


requirements = [
    'enaml',
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
]


setup(
    name='cochleogram',
    version='0.0.1',
    author='Brad Buran',
    author_email='bburan@alum.mit.edu',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE.txt',
    description='Module for creating cochleograms from confocal images',
    entry_points={
        'console_scripts': [
            'cochleogram=cochleogram.main:main',
        ],
    },
)
