import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from tqdm import tqdm
import copy
import logging
import argparse
from utils.data import FederatedDataset, load_data_for_domain, generate_nico_dg_train_ours, generate_nico_unique_train_ours
from utils.models import initialize_model
from utils.logger import setup_logger
# Argument parser to select backbone model
parser = argparse.ArgumentParser(description="Train a model on the NICO++_DG dataset")
parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'densenet121', 'vit_b_16', 'vit_b_32'], help='Backbone model to use')
parser.add_argument('--dataset', type=str, default='nico_dg', choices=['nico_dg', 'nico_u', 'domainnet'], help='Dataset for Training')
parser.add_argument('--round', type=int, default=1, help='Round')


args = parser.parse_args()
number_of_images = 10
# Set up logging

# Define the base directory for log files
logger_base_dir = "logs_extra/ours_30_images/"+ args.backbone + "_" + args.dataset + "_" + str(args.round)
os.makedirs(logger_base_dir, exist_ok=True)


# Example usage
# Global loss logger, accuracy logger and client loss logger
gloss_logger = setup_logger('gloss_logger', os.path.join(logger_base_dir, 'global_loss.log'), level=logging.INFO)
accuracy_logger = setup_logger('accuracy_logger', os.path.join(logger_base_dir, 'accuracy.log'), level=logging.INFO)
closs_logger = setup_logger('closs_logger', os.path.join(logger_base_dir, 'client_loss.log'), level=logging.INFO)


# Define base paths
base_path = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/"
if args.dataset == "nico_dg":
    test_base_path = os.path.join(base_path, "NICO_DG_official")
    test_domains = ["outdoor", "autumn", "dim", "grass", "rock", "water"]
    combined_test_file = os.path.join(base_path, "combined_test_files.txt")
    train_base_path = os.path.join(base_path, "NICO_DG")
elif args.dataset == "nico_u":
    file_base_path = os.path.join(base_path, "NICO_unique_official", "NICO_unique_official")

else:
    raise ValueError("Dataset Not Implemented")

# Initialize category dictionary
category_dict = {}
# Read test files and create category dictionary

### -------- Training file text file generation ------###
# Define training base path

# Initialize training file list

# Data transforms
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




class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.samples = []
        self.transform = transform

        # Read the file and parse image paths and labels
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                # Split from the right side, only at the last space
                image_path, label = line.rsplit(' ', 1)
                # print(f"The file image_path is {image_path} and Label is {label}")
                self.samples.append((image_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomDatasetUnique(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ---- If want to generate the NICO_DG File
# generate_nico_dg_train_ours()

if args.dataset == "nico_dg":
    train_file_path = os.path.join(base_path, "train_files.txt")
    # Training Dataset
    image_datasets = {
        'train': CustomDataset(train_file_path, data_transforms['train']),
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    }

    clients_test_data = {domain: load_data_for_domain(domain, is_train=False) for domain in test_domains}
    clients_test_datasets = {domain: FederatedDataset(data, data_transforms['test']) for domain, data in clients_test_data.items()}
    clients_test_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True) for domain, dataset in clients_test_datasets.items()}
elif args.dataset == "nico_u":
    category_dict, training_data = generate_nico_unique_train_ours()
    clients_test_data = generate_nico_unique_train_ours(is_train=False)
    image_datasets = {
        'train': CustomDatasetUnique(training_data, data_transforms['train'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4)
    }
    clients_test_datasets = {client: FederatedDataset(data, data_transforms['test']) for client, data in clients_test_data.items()}
    clients_test_dataloaders = {client: DataLoader(dataset, batch_size=32, shuffle=True) for client, dataset in clients_test_datasets.items()}

else:
    raise ValueError("Dataset Not Implemented")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to train the model without validation dataset
def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # logger.info('-' * 10)

        # Set model to training mode
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                # print(f"Outputs shape: {outputs.shape}")
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass + optimize
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        gloss_logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Deep copy the model if it's the best so far
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print('Early stopping')
            # logger.info('Early stopping')
            model.load_state_dict(best_model_wts)
            return model

        scheduler.step(epoch_loss)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Function to test the model
def test_model(model, test_dataloaders):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device model is on
    total_correct = 0
    total_samples = 0
    
    domain_accuracies = {}
    
    with torch.no_grad():
        for domain, dataloader in test_dataloaders.items():
            domain_correct = 0
            domain_samples = 0
            
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                # loss = nn.CrossEntropyLoss(outputs, labels)
                domain_correct += correct
                domain_samples += labels.size(0)
            
            domain_accuracy = 100 * domain_correct / domain_samples
            domain_accuracies[domain] = domain_accuracy
            total_correct += domain_correct
            total_samples += domain_samples
            
            print(f"Accuracy for domain {domain}: {domain_accuracy:.2f}%")
            accuracy_logger.info(f"Accuracy for domain {domain}: {domain_accuracy:.2f}%")
    
    overall_accuracy = 100 * total_correct / total_samples
    print(f"Overall accuracy across all domains: {overall_accuracy:.2f}%")
    accuracy_logger.info(f"Overall accuracy across all domains: {overall_accuracy:.2f}%")
    return domain_accuracies, overall_accuracy


# Main function
if __name__ == '__main__':
    # Initialize the model
    # print(f"Length of category dict is :{category_dict}")
    model = initialize_model(args.backbone, 60, pretrained=True)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

    # Test the model
    test_model(model, clients_test_dataloaders)