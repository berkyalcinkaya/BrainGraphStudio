from setuptools import setup

requires = [
        'numpy~=1.21.2',
        'pandas~=1.2.3',
        'scikit-learn~=1.0.2',
        'networkx~=2.6.2',
        'node2vec~=0.4.4',
        'scipy~=1.7.3',
        'nni~=2.4'
    ]

extras_require={
        'cpu': [
            'torch==1.8.0',
            'torchvision==0.9.1',
            'torchaudio==0.8.1'
        ],
        'gpu': [
            'torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
            'torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
            'torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html'
        ]
    }

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = ["BrainGraphStudio", "BrainGraphStudio.gui", "BrainGraphStudio.gui.ims",
            "BrainGraphStudio.models", "BrainGraphStudio.models.brainGNN", "BrainGraphStudio.BrainGB",
            "BrainGraphStudio.src", "BrainGraphStudio.src.dataset", "BrainGraphStudio.src.models", 
            "BrainGraphStudio.src.utils"]

setup(
    name = "BrainGraphStudio",
    version = "0.1.0",
    description = "A GUI-based toolkit for building, training, and optimizing graph neural networks for brain graph analysis",
    author = "Berk Yalcinkaya",
    url = "https://github.com/berkyalcinkaya/BrainGraphStudio",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email="berkyalcinkaya55@gmail.com",
    license = "BSD",
    packages = packages,
    install_requires = requires,
    extras_require=extras_require,
    classifiers=(
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
    entry_points = {
        'console_scripts': [
          'bgs = BrainGraphStudio.__main__:main']
       }
)