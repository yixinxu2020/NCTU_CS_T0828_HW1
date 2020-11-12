# NCTU_CS_T0828_HW1-cars image classification
## Introduction
The proposed challenge is a car images classification task using the Stanford car dataset, which contains 196 classes.
Train dataset | Test dataset
------------ | ------------- |
11,185 images | 5,000 images
## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-7500U CPU @ 2.90GHz
- 2x NVIDIA 2080Ti
## Data pre-process
### Data_classes
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
    |   +- label 3 ....(total 196 species labels )
```
### Data augmentation
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
## Training
### Model architecture
PyTorch provides several pre-trained models with different architectures. 

Among them, **ResNet152** is the architecture I adopted and I redefine the last layer to output 196 values, one for each class. As it gave the best validation accuracy upon training on our data, after running various architectures for about 10 epochs.
### Train models
To train models, run following commands.
```
$ python3 training.py
```
### Hyperparameters
Batch size=32, epochs=20
#### Loss Functions
Use the nn.**CrossEntropyLoss()**
#### Optimizer
Use the **SGD** optimizer with lr=0.01, momentum=0.9.

Use the lr_scheduler.**StepLR()** with step_size=10, gamma=0.1 to adjust learning rate. 
### Plot the training_result
Use the **plot.py** to plot the training_result

---training_data loss & acc, and validation_data loss & acc
```
$ python3 plot.py
```
## Testing
Using the trained model to pred the testing_data.
```
$ python3 test_data.py
```
And get the result, save it as csv file.
## Submission
Submit the test_result csv file, get the score.

The best resutl is 91.92%.


