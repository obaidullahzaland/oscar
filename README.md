# One Shot Federated Learning with Classifier-Free Diffusion Models -- ICME 2025
## This repository is the official Pytorch implementation for the paper "One Shot Federated Learning with Classifier-Free Diffusion Models," accepted at the International Conference on Multimedia and Expo (ICME 2025).

### Link to the Paper:   [![arXiv](https://img.shields.io/badge/arXiv-2502.08488-b31b1b.svg)](https://arxiv.org/abs/2502.08488) 

In order to run the repository, clone the repository:

```
https://github.com/obaidullahzaland/oscar.git
```
Install the requirements 
```
pip install -r requirements.txt
```
To generate captions
```
python caption_generator.py --dataset "/path/to/dataset"
```
Generate the encodings and images for the dataset
```
python infer_batch.py --category_path "/path/to/dataset/category" --standard_category "category_name" --sub_category "sub_category_name"
```

Run the training script
```
sh train_ours.sh
```
