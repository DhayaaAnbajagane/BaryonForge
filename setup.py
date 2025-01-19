from setuptools import setup, find_packages

setup(
    name='BaryonForge',
    version='1.1.0',
    author='Dhayaa Anbajagane',
    author_email='dhayaa@uchicago.edu',
    description='A pipeline for adding baryonic imprints and thermodynamic maps to N-body simulations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DhayaaAnbajagane/BaryonForge',
    packages=find_packages(),  # Automatically find packages in your project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'tqdm',
        'healpy',
        'scipy>=1.12',
        'numpy==1.*',
        'numba',
        'pyccl>=3.1.2'
    ],
    entry_points={
        'console_scripts': [
            # If you have a command-line tool, you can define it here
            # 'command-name=module:function',
        ],
    },
    include_package_data = False,  # Include non-Python files specified in MANIFEST.in
    zip_safe = False,
)
