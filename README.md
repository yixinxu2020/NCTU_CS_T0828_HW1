# NCTU_CS_T0828_HW1-cars image classification
## Introduction
The proposed challenge is a car images classification task using the Stanford car dataset, which contains 196 classes.
Train dataset | Test dataset
------------ | ------------- |
11,185 images | 5,000 images
## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
- 3x NVIDIA TitanX
## Methodology
### Data pre-process
#### Data_classes
All required files except images are already in data directory.
Using dataclass.py to devide the training_data into cars’ categories according to the labels from the csv file.
$ python3 dataclass.py

