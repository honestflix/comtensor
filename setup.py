from setuptools import setup, find_packages

setup(
    name='comtensor',
    version='0.1.0',
    author='Your Name',
    author_email='gitphantom@gmail.com',
    description='Bridging com and tao',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Comtensor/comtensor',
    packages=find_packages(),
    install_requires=[
        'bittensor'
    ],
    classifiers=[
        # Choose your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
