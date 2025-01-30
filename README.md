
ConceptAttention is an interpretability method for mulit-modal diffusion transformers. We implement it for the Flux DiT architecture in pytorch. 

<!-- Figure here -->


# Code setup

You will need to install all of the dependencies specified in `requirements.txt`. You will then need to install the code here locally by running
```bash
    pip install -e .
```
# Experiments

Each of our experiments are in separate directories in `experiments`. 

# Data Setup
To use ImageNetSegmentation you will need to download `gtsegs_ijcv.mat` into `experiments/imagenet_segmentation/data`. 

```bash
    cd experiments/imagenet_segmentation/data
    wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
```