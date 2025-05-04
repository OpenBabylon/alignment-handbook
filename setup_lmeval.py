from setuptools import setup, find_packages

setup(
    name='lm-eval-script',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'lm-eval @ git+https://github.com/PolyAgent/lm-evaluation-harness'
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'lm-eval-script=your_script_name:main',
        ],
    },
    author='Your Name',
    description='A script to evaluate language models using lm_eval',
    python_requires='>=3.8'
)
