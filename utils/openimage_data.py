import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import torch

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

root_dir = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/openimage"

class H5Dataset(Dataset):

    def __init__(self, num_samples, transforms=None):
        self.transforms = transforms
        self.data = []
        self.labels = []
        for i in range(6):  # Replace with your HDF5 file path
            domain_path = os.path.join(root_dir, "train", f"client_{i}")
            domain_file_path = os.path.join(domain_path, "generated_images.h5")
            print(f"Processing client {i} at {domain_file_path} which is a file {os.path.exists(domain_file_path)}")

            with h5py.File(domain_file_path, 'r') as h5_data:
                for dataset_name in os.listdir(domain_path):
                    if dataset_name.endswith("h5"):
                        continue
                    dataset = h5_data[str(dataset_name)]
                    # Iterate through each image and its label
                    count = 0
                    for image in dataset:
                        if count < num_samples:
                            size = image.shape
                            self.data.append(image)
                            self.labels.append(int(dataset_name))
                            count = count+1


        squeezed_image = np.squeeze(image, axis=0)
        print(f"Shape is {size} Squeezed shape is {np.squeeze(image, axis=0).shape} and Reshaped is {np.transpose(squeezed_image, (2, 0, 1)).shape}")
        # Convert data and labels to numpy arrays for efficiency
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single image and its corresponding label
        image = self.data[idx]
        label = self.labels[idx]
        image = np.squeeze(image, axis=0)  # Remove the extra dimension
        # image = np.transpose(image, (2, 0, 1))  # Rearrange to [C, H, W]
        image = Image.fromarray(image)

        # Convert image and label to PyTorch tensors
        if self.transforms:
            image = self.transforms(image)

        return image, label



def generate_openimage_ours(num_samples = 10):  
    print("Generating Dataset")  
    dataset = H5Dataset(num_samples, transforms=data_transforms['train'])
    print(f"Lenght of dataset is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, dataloader


def create_data_list(client_id, data_type, samples=10):
    data_list = []
    client_path = os.path.join(root_dir, data_type, f"client_{str(client_id)}")
    print(f"{os.path.isdir(client_path)}")
    print(f"{client_path}")
    for category in range(20):  # Categories are named from 0 to 19
        category_path = os.path.join(client_path, str(category))
        data_size = 0
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                if data_type == 'train' and data_size > samples:
                    continue
                image_path = os.path.join(category_path, image_name)
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data_list.append((image_path, category)) 
                    data_size += 1 # 
    return data_list

def load_federated_data_openimage(samples=10):
    train_data = {}
    test_data = {}
    for client_id in range(6):  # Clients are named from client_0 to client_5
        train_data_list = create_data_list(client_id, "train", samples=samples)
        test_data_list = create_data_list(client_id, "test", samples=samples)
        # print(f"The first few files for client are {train_data_list[0]}, {train_data_list[1]}, {train_data_list[2]}")       
        print(f"Lenght of data is : {len(train_data_list)}")
        train_data[f"client_{client_id}"] = train_data_list
        test_data[f"client_{client_id}"] = test_data_list
        print(f"Loaded train and test datasets for client_{client_id}.")

    return train_data, test_data


