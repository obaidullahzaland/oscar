import os
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import argparse
import h5py

os.environ['TORCH_HOME'] = '/torch/cache'
os.environ['HF_HOME'] = '/HF/home'
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", help="Root directory for files")
args = parser.parse_args()

def generate_and_save_captions_single_file(root_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)


    if os.path.isdir(root_dir):
        print("Cateogry Path Accessed")
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                captions_dir = os.path.join(category_path, 'captions')
                captions_file = os.path.join(captions_dir, "captions.txt")
                os.makedirs(captions_dir, exist_ok=True)
                
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        raw_image = Image.open(image_path).convert('RGB')
                        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                        captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
                        
                        # Save captions to a text file
                        # caption_file_path = os.path.join(captions_dir, f"{os.path.splitext(image_name)[0]}.txt")
                        with open(captions_file, 'w') as caption_file:
                            for caption in captions:
                                caption_file.write(f"{image_name} {caption}")
                        # print(f"Captions for {image_name} saved to {caption_file_path}")
    else:
        print("The root file is not a directory")


def generate_and_save_captions(root_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)

    for category in os.listdir(root_dir):
        if category == "text_files":
            continue
        category_path = os.path.join(root_dir, category)
        
        if os.path.isdir(category_path):
            for domain in os.listdir(category_path):
                domain_path = os.path.join(category_path, domain)
                if os.path.isdir(domain_path):
                    # Create the captions directory if it doesn't exist
                    captions_dir = os.path.join(domain_path, 'captions')
                    os.makedirs(captions_dir, exist_ok=True)
                    print("Caption Dir made")
                    for image_name in os.listdir(domain_path):
                        image_path = os.path.join(domain_path, image_name)
                        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            raw_image = Image.open(image_path).convert('RGB')
                            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
                            
                            # Save captions to a text file
                            caption_file_path = os.path.join(captions_dir, f"{os.path.splitext(image_name)[0]}.txt")
                            with open(caption_file_path, 'w') as caption_file:
                                for caption in captions:
                                    caption_file.write(caption + '\n')
                            # print(f"Captions for {image_name} saved to {caption_file_path}")


def generate_captions_hdf5(root_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
    )
    hdf5_file_path = os.path.join(root_dir, "captions.h5")
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        for category in os.listdir(root_dir):
            print(f"Category :{category}")
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # Create a group for each category in HDF5
                category_group = hdf5_file.create_group(category)
                
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        raw_image = Image.open(image_path).convert('RGB')
                        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                        
                        captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
                        
                        # Store the caption in the HDF5 file
                        dataset_name = os.path.splitext(image_name)[0]
                        category_group.create_dataset(dataset_name, data=captions[0])
                        
                        # print(f"Caption for {image_name} saved in HDF5 under category {category}.")



def read_captions(root_dir):
    hdf5_file = os.path.join(root_dir, "captions.h5")
    with h5py.File(hdf5_file, 'r') as hdf5_file:
        for category in hdf5_file.keys():
            print(f"Category: {category}")
            category_group = hdf5_file[category]
            
            for image_name in category_group.keys():
                caption = category_group[image_name][()].decode('utf-8')  # Decode byte string to text
                print(f"  Image: {image_name}, Caption: {caption}")


    
    



if __name__ == "__main__":
    generate_captions_hdf5(args.root_dir)


