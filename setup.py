from setuptools import find_packages, setup


requirements = [
    'aicspylibczi',
    'atom',
    'enaml',
    'matplotlib',
    'numpy',
    'raster_geometry',
    'pandas',
    'scipy',
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
            'cochleogram-prepare=cochleogram.main:main_prepare',
        ],
    },
)
