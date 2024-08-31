import model.stable_diffusion.generation as generation
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import h5py
import torch


def read_clip_embeddings(directory):
    embeddings = {}
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            with h5py.File(file_path, 'r') as hdf:
                data = hdf['embeddings'][:]
                embeddings[filename] = data
                array = embeddings[filename]
                embeddings_tensor = torch.tensor(array, dtype=torch.float16)
                # embeddings_tensor = embeddings_tensor.unsqueeze(0)
                # mean_embeddings = torch.mean(embeddings_tensor, dim=0)
                # mean_embeddings.unsqueeze(0)
    return embeddings_tensor
            

def process_image(standard_category=None, sub_category=None, category_path=None, num_images=None, save_path=None):
        for i in range(num_images):
            seed = np.random.randint(0, 10000)
            sub_category_embedding_path = category_path + standard_category + "/" + sub_category + "/SD_encodings/"
            sub_category_embedding = read_clip_embeddings(sub_category_embedding_path)
            # print(sub_category_embedding.shape)
            generation.generate(standard_category=standard_category, sub_category=sub_category, sub_category_embedding=sub_category_embedding, seed=seed, save_path = save_path)



def main(args):
    process_image(standard_category=args.standard_category, sub_category=args.sub_category, category_path=args.category_path, num_images=args.num_images, save_path=args.save_path, )
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs for the image generation script.")
    parser.add_argument('--standard_category', type=str, default="bear", help='standard category')
    parser.add_argument('--sub_category', type=str, default="black", help='sub category under the standar category')
    parser.add_argument('--category_path', type=str, default="/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/NICO_unique/", help='path to the dataset folder')    
    parser.add_argument('--save_path', type=str, default="/proj/cloudrobotics-nest/users/NICO++/Generation/bear_test", help='Path where the generated images will be saved')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    args = parser.parse_args()
    main(args)