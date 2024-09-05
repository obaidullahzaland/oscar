import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob


client_data_size = 10
ours_data_limit = 10
base_path = "/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/"
nico_unique_base_path = os.path.join(base_path, "NICO_unique")
test_base_path_DG = os.path.join(base_path, "NICO_DG_official")

test_domains = ["outdoor", "autumn", "dim", "grass", "rock", "water"]
category_dict = {}
train_file_list = []
train_base_path_DG = os.path.join(base_path, "NICO_DG")


def generate_nico_dg_train_ours():

    # Traverse the training directories and generate training file list
    for domain in test_domains:
        domain_path = os.path.join(train_base_path_DG, domain)
        for category_name, category_value in category_dict.items():
            category_path = os.path.join(domain_path, category_name, "generated_images")
            # print(f"The Cateogry path for Training is {category_path}")
            image_files = glob.glob(os.path.join(category_path, "*.png"))
            for image_file in image_files:
                train_file_list.append(f"{image_file} {category_value}")
    train_file_txt = os.path.join(base_path, "train_files.txt")
    with open(train_file_txt, 'w') as f:
        for item in train_file_list:
            f.write(f"{item}\n")


def read_data(file_path, nico_u_base):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            parts = line.strip().split('/')
            # print(parts)
            b = parts[-1].split(' ')
            cat_num = b[1]
            file_name = b[0]
            cat = parts[5]
            dom = parts[6]
            if cat not in category_dict:
                category_dict[cat] = cat_num 
            full_file_path = os.path.join(nico_u_base, cat, dom, file_name)
            if ".DS_Store" in full_file_path:
                pass
            else:
                data.append((full_file_path, int(cat_num)))
            # print(data)
            # full_path = os.path.join()
        return data
        

def generate_nico_unique_train_ours(is_train=True, method='ours'):
    nico_files_base_path = os.path.join(base_path, "NICO_unique_official", "NICO_unique_official")
    nico_data_base_path = os.path.join(base_path, "NICO_unique")
    federated_train = {f"client_{i+1}": [] for i in range(6)}
    federated_test = {f"client_{i+1}": [] for i in range(6)}
    c_size = {}
    centralized_train = []
    ours_train = []
    category_assignment = {}

    for entry in os.listdir(nico_files_base_path): 
        
        file_path = os.path.join(nico_files_base_path, entry)
        # print(file_path)
        
        if os.path.isfile(file_path):
            # print(file_path)
            parts = entry.split("_")
            # print(parts)
            category, domain, _ = parts[0], parts[1], parts[2]
            # print(f"{category} and {domain} and {_}")
            # Assigning domains to clients
            if category not in category_assignment:
                category_assignment[category] = 1
            # print(category_assignment)
            data = read_data(file_path, nico_data_base_path)
            if is_train:
                if len(data) > client_data_size:
                    data = data [:client_data_size]
            # print(data)
            client_id = f"client_{category_assignment[category]}"
            if is_train and 'train.txt' in entry:
                # print(f"Category {category} and Domain {domain}")
                federated_train[client_id].extend(data)
                centralized_train.extend(data)
                category_assignment[category] += 1
            elif not is_train and 'test.txt' in entry:
                federated_test[client_id].extend(data)
                category_assignment[category] += 1

    if method=='ours':
        for c in category_dict.keys():
            c_path = os.path.join(nico_data_base_path, c)
            domains = os.listdir(c_path)
            for domain in domains:
                category_path = os.path.join(c_path, domain, "generated_images")
                image_files = glob.glob(os.path.join(category_path, "*.png"))
                if len(image_files) > ours_data_limit:
                    image_files = image_files[:ours_data_limit]
                for image_file in image_files:
                    ours_train.append((image_file, int(category_dict[c])))

    if method == 'ours':
        if is_train:
            return category_dict, ours_train
        else:
            return federated_test
    elif method == 'federated':
        if is_train:
            return category_dict, federated_train
        else:
            return federated_test
    elif method == 'centralized':
        if is_train:
            return category_dict, centralized_train
        else:
            return federated_test



        

def load_nico_dg_centralized(domains):
    combined_test_file = os.path.join(base_path, "combined_test_files.txt")
    combined_train_file = os.path.join(base_path, "combined_train_file.txt")
    test_file_paths = []
    train_file_paths = []
    with open(combined_test_file, 'w') as combined_file:
        for domain in domains:
            test_file = os.path.join(test_base_path_DG, f"{domain}_test.txt")
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
                    if category_name not in category_dict:
                        category_dict[category_name] = category
                    full_file_path = os.path.join(base_path, "NICO_DG", domain_name, category_name, file_name)
                    combined_file.write(f"{full_file_path} {category}\n")


    with open(combined_train_file, 'w') as combined_t_file:
        for domain in domains:
            train_file = os.path.join(test_base_path_DG, f"{domain}_train.txt")
            with open(train_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    b = line.split('/')
                    category_name = b[2]
                    category =  int(b[-1].split(" ")[1])
                    domain_name = b[1]
                    file_name = b[-1].split(" ")[0]
                    if category_name not in category_dict:
                        category_dict[category_name] = category
                    full_file_path = os.path.join(base_path, "NICO_DG", domain_name, category_name, file_name)
                    combined_t_file.write(f"{full_file_path} {category}\n")


    return combined_train_file, combined_test_file, category_dict



def load_data_for_domain(domain, is_train=True):
    file_suffix = 'train.txt' if is_train else 'test.txt'
    file_path = os.path.join(test_base_path_DG, f"{domain}_{file_suffix}")
    data_list = []
    with open(file_path, 'r') as file:
        c_size = {}
        lines = file.readlines()
        for line in lines:
            b = line.split('/')
            category_name = b[2]
            domain_name = b[1]
            category =  int(b[-1].split(" ")[1])
            file_name = b[-1].split(" ")[0]
            if category_name not in category_dict:
                category_dict[category_name] = category
            if category_name not in c_size:
                c_size[category_name] = 1
            if is_train:
                if c_size[category_name] <= client_data_size:
                    full_file_path = os.path.join(base_path, "NICO_DG", domain_name, category_name, file_name)
                    data_list.append((full_file_path, category))
                    c_size[category_name] += 1
            else:
                full_file_path = os.path.join(base_path, "NICO_DG", domain_name, category_name, file_name)
                data_list.append((full_file_path, category))
    return data_list




    # Federated dataset setup
class FederatedDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.samples = [(path, int(label)) for path, label in data_list]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


