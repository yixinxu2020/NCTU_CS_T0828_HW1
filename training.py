from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import os
import time
import copy

# set up the running device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# using 450x300 images with random horizontal flip, random rotation and normalization,
# and transform img to tensor
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((450, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((450, 300)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
# Encapsulate the pre-processed data by DataLoader
data_dic = "./data1/"
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dic, x), data_transforms[x])
                  for x in ['train', 'val']}
data_loaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


# define a train model
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    # set the weights and initial acc
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    training_losses = []
    training_accs = []
    test_losses = []
    test_accs = []

    # Each epoch has a training and validation phase
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to eval mode

            running_loss = 0.0

            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                # Get the data and send it to CUDA
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            # calculate loss and acc
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                epoch_loss.append(training_losses)
                epoch_acc.append(training_accs)
                print(training_losses, training_accs)
            elif phase == 'val':
                epoch_loss.append(test_losses)
                epoch_acc.append(test_accs)
                print(test_losses, test_accs)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model, update weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training cpmplete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, training_losses, training_accs, test_losses, test_accs


def main():
    # use pre_trained resnet50model
    model_ft = models.resnet152(pretrained=True)
    # replace the last fc layer with an untrained one (requires grad by default)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 196)
    # sent model to CUDA
    model_ft = model_ft.to(device)
    # set criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    model_ft, training_losses, training_accs, test_losses, test_accs = \
        train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    torch.save(model_ft.state_dict(), './best_model152.pt')
    # plot the model result
    f, axarr = plt.subplots(2, 2, figsize=(12, 8))
    axarr[0, 0].plot(training_losses)
    axarr[0, 0].set_title("Training loss")
    axarr[0, 1].plot(training_accs)
    axarr[0, 1].set_title("Training acc")
    axarr[1, 1].plot(test_accs)
    axarr[1, 1].set_title("Test acc")
    axarr[1, 0].plot(test_losses)
    axarr[1, 0].set_title("Training loss")


if __name__ == '__main__':
    main()
