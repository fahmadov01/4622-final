# 4622-final

# ImageTransformer

## Overview
`ImageTransformer.py` is a script for training an image classification transformer model using fully annotated images and their corresponding annotation files from Mapillary.

## Dataset Structure
To be able to run the script, please ensure you have the following project structure. The images can be downloaded from [mapillary.com](mapillary.com). Please make sure that you only get the fully annotated images along with the fully annotated annotations.

project_root/
│
├── images/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── annotations/
│ ├── annotation1.json
│ ├── annotation2.json
│ └── ...
│
└── imagetransformer.py

## Requirements

You will need a few libraries to be able to run the script. You can install the required dependencies using pip:

```bash
pip install torch torchvision transformers pillow numpy
```

To run the program, you can simply run python3 ImageTransformers.py. Please ensure you are using Python version 3.12.
