import os
from torchvision import transforms
from PIL import Image
import glob
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



client_data_size = 10 # This is for FL methods
ours_data_limit = 40 # This is for our method
base_path = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/"

domainnet_base = os.path.join(base_path, "domainNet")
domainnet_domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
train_file_list = []
domainnet_categories = {}
domainnet_test_path = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/domainNet/text_files/"
domainnet_train_path = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/domainNet/text_files/"


data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224,224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
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



class H5Dataset(Dataset):

    def __init__(self, label_dict, num_samples, transforms=None):
        self.transforms = transforms
        self.data = []
        self.labels = []
        for domain in domainnet_domains:  # Replace with your HDF5 file path
            domain_file_path = os.path.join(domainnet_base, domain, "generated_images.h5")
            with h5py.File(domain_file_path, 'r') as h5_data:
                for dataset_name, label in label_dict.items():
                    dataset = h5_data[dataset_name]
                    # Iterate through each image and its label
                    count = 0
                    for image in dataset:
                        if count < num_samples:
                            self.data.append(image)
                            if int(label) > 90:
                                self.labels.append(int(label) - 190)
                            else:
                                self.labels.append(label)
                            count = count+1
                        else:
                            break

        
        # Convert data and labels to numpy arrays for efficiency
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Labels max is {np.max(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single image and its corresponding label
        image = self.data[idx]
        label = self.labels[idx]
        image = np.squeeze(image, axis=0)  # Remove the extra dimension
        # image = np.transpose(image, (2, 0, 1))  # Rearrange to [C, H, W]
        image = Image.fromarray(image)
        if self.transforms:
            image = self.transforms(image)
        # Convert image and label to PyTorch tensors
        # image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        # label = torch.tensor(label, dtype=torch.long)

        return image, label


def create_label_dict():
    label_dict = {}
    for domain in domainnet_domains:
        sample_file_path = os.path.join(domainnet_train_path, f"{domain}_train.txt")
        with open(sample_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                b = line.split('/')
                category_name = b[1]
                category =  int(b[-1].split(" ")[1])
                if int(category)<30:
                    if category_name not in label_dict:
                        label_dict[category_name] = category
                elif int(category)>=40 and int(category) < 90:
                    if category_name not in label_dict:
                        label_dict[category_name] = category
                elif int(category)>=220 and int(category)<230:
                    if category_name not in label_dict:
                        label_dict[category_name] = category
    return label_dict

def generate_domainnet_ours(num_samples = 10):
    label_dict = create_label_dict()
    print(len(label_dict))
    dataset = H5Dataset(label_dict, num_samples, transforms=data_transforms['train'])
    print(f"Lenght of dataset is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    return dataset, dataloader

# Example: Iterate through DataLoader
# for batch_images, batch_labels in dataloader:
#     print("Batch images shape:", batch_images.shape)  # e.g., [8, 512, 512, 3]
#     print("Batch labels:", batch_labels)
#     break



# def generate_domainnet_train_ours():
#     # Traverse the training directories and generate training file list
#     for domain in domainnet_domains:
#             train_file = os.path.join(domainnet_train_path, f"{domain}_train.txt")
#             with open(train_file, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     line = line.strip()
#                     b = line.split('/')
#                     category_name = b[2]
#                     category =  int(b[-1].split(" ")[1])
#                     domain_name = b[1]
#                     file_name = b[-1].split(" ")[0]
#                     if category_name not in domainnet_categories:
#                         domainnet_categories[category_name] = category
#     for domain in domainnet_domains:
#         domain_path = os.path.join(domainnet_base, domain)
#         for category_name, category_value in domainnet_categories.items():
#             category_path = os.path.join(domain_path, category_name, "generated_images")
#             # print(f"The Cateogry path for Training is {category_path}")
#             image_files = glob.glob(os.path.join(category_path, "*.png"))
#             if len(image_files > ours_data_limit):
#                 image_files = image_files[:ours_data_limit]
#             for image_file in image_files:
#                 train_file_list.append((image_file, category_value))
#     return train_file_list

        


def load_domainnet_centralized_domainnet():
    combined_test_file = os.path.join(base_path, "domainnet_test_files.txt")
    combined_train_file = os.path.join(base_path, "domainnet_train_file.txt")
    test_file_paths = []
    train_file_paths = []
    with open(combined_test_file, 'w') as combined_file:
        for domain in domainnet_domains:
            test_file = os.path.join(domainnet_test_path, f"{domain}_test.txt")
            with open(test_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # print(line)
                    line = line.strip()
                    b = line.split('/')
                    category_name = b[2]
                    category =  int(b[-1].split(" ")[1])
                    domain_name = b[1]
                    file_name = b[-1].split(" ")[0]
                    if category_name not in domainnet_categories:
                        domainnet_categories[category_name] = category
                    full_file_path = os.path.join(base_path, "domaiNnet", domain_name, category_name, file_name)
                    combined_file.write(f"{full_file_path} {category}\n")


    with open(combined_train_file, 'w') as combined_t_file:
        for domain in domainnet_domains:
            train_file = os.path.join(domainnet_train_path, f"{domain}_train.txt")
            with open(train_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    b = line.split('/')
                    category_name = b[2]
                    category =  int(b[-1].split(" ")[1])
                    domain_name = b[1]
                    file_name = b[-1].split(" ")[0]
                    if category_name not in domainnet_categories:
                        domainnet_categories[category_name] = category
                    full_file_path = os.path.join(base_path, "domaiNnet", domain_name, category_name, file_name)
                    combined_t_file.write(f"{full_file_path} {category}\n")

    return combined_train_file, combined_test_file, domainnet_categories



def load_data_for_domain_domainnet(domain, is_train=True, samples=10):
    label_dict = create_label_dict()
    file_suffix = 'train.txt' if is_train else 'test.txt'
    file_path = os.path.join(domainnet_train_path, f"{domain}_{file_suffix}")
    data_list = []
    with open(file_path, 'r') as file:
        is_file = 0
        is_not_file = 0
        c_size = {}
        lines = file.readlines()
        for line in lines:
            b = line.split('/')
            category_name = b[1]
            domain_name = b[0]
            category =  int(b[-1].split(" ")[1])
            file_name = b[-1].split(" ")[0]
            # print(f"File Path {file_name} domain Name {domain_name} Category {category} Category Name {category_name}")
            if category_name in label_dict:
                if category_name not in domainnet_categories:
                    domainnet_categories[category_name] = category
                if category_name not in c_size:
                    c_size[category_name] = 1
                if is_train:
                    if c_size[category_name] <= samples:
                        full_file_path = os.path.join(base_path, "domainNet", domain_name, category_name, file_name)
                        if int(category)<30:
                            data_list.append((full_file_path, category))
                        elif int(category) >=40 and int(category) < 90:
                            data_list.append((full_file_path, category))
                        elif int(category)>=220 and int(category)<230:
                            data_list.append((full_file_path, int(category)-190))
                            c_size[category_name] += 1
                else:
                    full_file_path = os.path.join(base_path, "domainNet", domain_name, category_name, file_name)
                    if int(category)<30:
                        data_list.append((full_file_path, category))
                    elif int(category) >=40 and int(category) < 90:
                        data_list.append((full_file_path, category))
                    elif int(category)>=220 and int(category)<230:
                        data_list.append((full_file_path, int(category)-190))
                
                # count = count+1
                    # data_list.append((full_file_path, category))
    print(f"File {is_file} Not File {is_not_file}")
    return data_list




#### Domain Net data






