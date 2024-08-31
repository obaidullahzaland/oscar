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
from utils.data import FederatedDataset, generate_nico_unique_train_ours, load_data_for_domain, load_nico_dg_centralized
from utils.logger import setup_logger
from utils.models import initialize_model

parser = argparse.ArgumentParser(description="Train a model on the NICO++_DG dataset")
parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'densenet121', 'vit_b_16', 'vit_b_32'], help='Backbone model to use')
parser.add_argument('--round', type=int, default=1, help='Round')
parser.add_argument('--dataset', type=str, default='nico_dg', choices=['nico_dg', 'nico_u', 'domainnet'], help='Dataset for Training')

args = parser.parse_args()


number_of_images = 10
# Define the base directory for log files
logger_base_dir = "logs_extra/centralized_"+ args.backbone + "_" + args.dataset + "_" + str(args.round)
os.makedirs(logger_base_dir, exist_ok=True)


# Example usage
# Global loss logger, accuracy logger and client loss logger
logger = setup_logger('logger', os.path.join(logger_base_dir, 'logger.log'), level=logging.INFO)


base_path = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/"

test_domains = ["outdoor", "autumn", "dim", "grass", "rock", "water"]
combined_train_file, combined_test_file, cateogry_dict = load_nico_dg_centralized(test_domains)


class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.samples = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                image_path, label = line.rsplit(' ', 1)
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
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if args.dataset == 'nico_dg':
    # Paths to text files
    train_file_path = combined_train_file
    # Assuming `test_file_paths` is a list of (file_path, label) tuples
    test_file_path = combined_test_file  # Create a test list file similarly

    # Create datasets
    image_datasets = {
        'train': CustomDataset(train_file_path, data_transforms['train']),
        'test': CustomDataset(test_file_path, data_transforms['test'])
    }

    # Create data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
    }

    clients_test_data = {domain: load_data_for_domain(domain, is_train=False) for domain in test_domains}
    clients_test_datasets = {domain: FederatedDataset(data, transform) for domain, data in clients_test_data.items()}
    clients_test_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True) for domain, dataset in clients_test_datasets.items()}
    # Define the device

elif args.dataset == 'nico_u':
    category_dict, centralized_train = generate_nico_unique_train_ours(is_train = True, method='centralized')
    clients_test_data = generate_nico_unique_train_ours(is_train = False, method='centralized')
    image_datasets = {
        'train': CustomDatasetUnique(centralized_train, data_transforms['train'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4)
    }
    clients_test_datasets = {client: FederatedDataset(data, data_transforms['test']) for client, data in clients_test_data.items()}
    clients_test_dataloaders = {client: DataLoader(dataset, batch_size=32, shuffle=True) for client, dataset in clients_test_datasets.items()}

else:
    raise ValueError("Dataset Not Implemented")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Centralized train 0 : {centralized_train[0]}")
# Function to train the model without validation dataset
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        logger.info('-' * 10)

        # Set model to training mode
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in tqdm(dataloaders['train']):
            # print(f"Type of inputs: {type(inputs)}, Type of labels: {type(labels)}")
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
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
        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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
            logger.info('Early stopping')
            model.load_state_dict(best_model_wts)
            return model

        scheduler.step(epoch_loss)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
                domain_correct += correct
                domain_samples += labels.size(0)
            
            domain_accuracy = 100 * domain_correct / domain_samples
            domain_accuracies[domain] = domain_accuracy
            total_correct += domain_correct
            total_samples += domain_samples
            
            print(f"Accuracy for domain {domain}: {domain_accuracy:.2f}%")
            logger.info(f"Accuracy for domain {domain}: {domain_accuracy:.2f}%")
    
    overall_accuracy = 100 * total_correct / total_samples
    print(f"Overall accuracy across all domains: {overall_accuracy:.2f}%")
    logger.info(f"Overall accuracy across all domains: {overall_accuracy:.2f}%")
    return domain_accuracies, overall_accuracy
# Function to initialize the model


# Main function
if __name__ == '__main__':
    # Initialize the model
    model = initialize_model(args.backbone, len(cateogry_dict), pretrained=True)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

    domain_accuracies, overall_accuracy = test_model(model, clients_test_dataloaders)
    print("Model Training Completed")
