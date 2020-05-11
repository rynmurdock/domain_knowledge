


# Is domain knowledge necessary for machine learning materials properties?

**Ryan Murdock, Steven K. Kauwe, Anthony Yu-Tung Wang & Taylor D. Sparks**

preprint doi: 

link to paper: 

## Setup
This code requires TensorFlow 1 and CUDA. Linux users will need to install CUDA and cuDNN separately.
### 1. Download or Clone the repository
https://github.com/rynmurdock/domain_knowledge.git

### 2. Install via anaconda:
1. Navigate to directory. Open anaconda prompt.
1. Run the following from anaconda prompt to create environment:
`conda env create --file conda-env.yml`
1. Run the following to activate environment:
`source activate data_is_everything`


--------
## Usage
### Get learning curves
1. Train models and save metrics using validation dataset.

    `python train_learning_curves.py`

2. Generate plots from saved metrics.

    `python figures/generate_learning_curves.py`

### Generate model weights and use for predictions


`python full_dataset_model_weights/generate_weights.py`

`python full_dataset_model_weights/test_with_weights.py`

### Generate r2 scores for different train/test splits using a ridge regression
###
 `python generate_test_splits_figure.py`

### Generate r2 scores of unseen element predictions with various descriptors using a ridge regression
###
 `python generate_holdouts_figure.py`
 `python generate_holdouts_heatmaps.py`
