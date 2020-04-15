from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='DISTS_pytorch',
    version='0.1',
    description='Deep Image Structure and Texture Similarity (DISTS) Metric',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['DISTS_pytorch'],
    package_data= {'': ['DISTS_pytorch/weights.pt']},
    data_files= [('', ['DISTS_pytorch/weights.pt'])],
    include_package_data=True,
    author='Keyan Ding',
    author_email='dingkeyan93@outlook.com',
    install_requires=["torch>=1.0"],
    url='https://github.com/dingkeyan93/DISTS',
    keywords = ['pytorch', 'similarity', 'VGG','texture','structure','metric'], 
    platforms = "python",
    license='MIT',
)

# python setup.py sdist bdist_wheel
# twine check dist/*
# twine upload dist/*