import os
import torch
import pandas as pd
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


# Make a list of img paths and labels list
file_path = './testing_data/testing_data/'
file_name = os.listdir(file_path)
img_path = [file_path + i for i in file_name]
dir_ = './data1/train'
labels = os.listdir(dir_)
labels_list = sorted(labels)
# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define test_dataset transform
test_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                            ])


# Customize a data entry function
class CarDataset(Dataset):
    def __init__(self, path, transform):
        self.img_path = path
        self.transform = transform

    def __getitem__(self, index):
        fn = self.img_path[index]
        img = Image.open(fn).convert("RGB")
        img = self.transform(img)
        return img, '1'

    def __len__(self):
        return len(self.img_path)


# Define the predictive function for test_data
def prediect(test_dataset):
    # Read the previously trained model
    net = models.resnet152(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 196)
    net.load_state_dict(torch.load('./best_model152.pt'))
    # trans the model to the server
    net.to(device)
    net.eval()
    # get the test_data and use the model to get the predicted results
    predition = []
    for inputs, _ in test_dataset:
        # Get the data and send it to CUDA
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            _, pred = torch.max(outputs, 1)
            for i in pred:
                predition.append(labels_list[i])
    # Get the id_num and label corresponding to the prediction result
    id_number = []
    for name in file_name:
        id_ = name.replace('.jpg', '')
        id_number.append(id_)
    result = {'id': id_number, 'label': predition}
    result = pd.DataFrame(result)
    # Save the results as a CSV file
    result.to_csv('resnet50_test_result.csv', index=False)


def main():
    # Dataset and DataLoader
    test_dataset = CarDataset(img_path, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, sampler=None, num_workers=0)

    prediect(test_loader)


if __name__ == '__main__':
    main()
