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

Using **dataclass.py** to devide the training_data into cars’ categories according to the labels from the csv file.
```
$ python3 dataclass.py
```
After deviding, the training_data becomes like this:
```
Trainin_data
  +- train
    |	+- label 1
    |	+- label 2
    | 	+- label 3 ....(total 196 species labels )
  +- val
    |	+- label 1
    |	+- label 2
    | +- label 3 ....(total 196 species labels )
```
#### Data augmentation
Since there are 196 kinds of cars to be trained, the training data may not be enough to cause overfit. Therefore, before input data into the model, we can generate more data for the machine to learn by means of data augmentation. 
```
transforms.Compose([
        transforms.Resize((450, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
### Model architecture
PyTorch provides several pre-trained models with different architectures. 

Among them, **ResNet152** is the architecture I adopted and I redefine the last layer to output 196 values, one for each class. As it gave the best validation accuracy upon training on our data, after running various architectures for about 10 epochs.
