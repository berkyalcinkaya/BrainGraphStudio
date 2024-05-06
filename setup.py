from matplotlib.pylab import geometric
from setuptools import setup

requires = [
    "brainGB"
    "nni",
    "numpy",
    "scipy",
    "matplotlib",
    "PyQt5==5.15.9",
    "scikit_image",
    "scikit_learn",
    "scipy"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = ["BrainGraphStudio", "BrainGraphStudio.gui", "BrainGraphStudio.gui.ims",
            "BrainGraphStudio.models", "BrainGraphStudio.models.brainGNN"]

setup(
    name = "BrainGraphStudio",
    version = "0.1.0",
    description = "A GUI-based toolkit for building, training, and optimizing graph neural networks for brain graph analysis",
    author = "Berk Yalcinkaya",
    url = "https://github.com/berkyalcinkaya/gnn_gui",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email="berkyalcinkaya55@gmail.com",
    license = "BSD",
    packages = packages,
    install_requires = requires,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
    entry_points = {
        'console_scripts': [
          'yeastvision = yeastvision.__main__:main']
       }
)

)