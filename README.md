# BNN\_for\_hyperspectral\_datasets\_analysis

This repository contains python code to train bayesian neural networks for some of the most widely used open hyperspectral imaging datasets and to analyse the results.

We will refer to the repository as `bnn4hi`. To clone it, it is recommended to change the folder destination, especially to use it as a module with the import clause:

> git clone https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis.git bnn4hi

This is the code of the paper *Bayesian Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty Metrics*. If it is useful to you, please cite:

A. Alcolea and J. Resano, "Bayesian Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty Metrics", in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-10, 2022, doi: 10.1109/TGRS.2022.3205119.

@article{alcolea2022bayesian,  
author = {Alcolea, Adrián and Resano, Javier},  
journal = {IEEE Transactions on Geoscience and Remote Sensing},  
title = {Bayesian Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty Metrics},  
year = {2022},  
volume = {60},  
pages = {1-10},  
doi = {10.1109/TGRS.2022.3205119}  
}

## How to run

### With the 'sh' script

> ./launch.sh

Will run every step needed to reproduce the experiments of the paper *Bayesian Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty Metrics*. Actually, it will only train the models for 100 epochs to easily test that everything works. There is a boolean variable in *launch.sh* called TEST\_EXECUTION set to *true*. For launching the same number of epochs of the paper, set it to *false*.

You must have python3 installed with the following libraries: numpy, matplotlib, spectral, scipy, scikit-learn, tensorflow and tensorflow\_probability.

### With docker

> docker build -t bnn4hi .

Will generate the container.

> docker run -v ${PWD}:/workdir bnn4hi

Will run *./launch.py* in docker using the repository directory as input and output point, so the result will be the same as running *./launch.py* but avoiding to install the dependencies.

## Use and files explanation

### Train

> ./train.py NAME EPOCHS PERIOD

Will train the NAME dataset for EPOCHS epochs saving a checkpoint and writing information to stdout every PERIOD epochs.

The resultant trained checkpoints will be in *Models/{NAME}\_{LAYER1\_NEURONS}-{LAYER2\_NEURONS}model\_{P\_TRAIN}train\_{LEARNING\_RATE}lr/*, organised with an *epoch\_{EPOCH}* directory for each checkpoint.

Where LAYER1\_NEURONS and LAYER2\_NEURONS correspond to the number of neurons of the layers of the model, P\_TRAIN corresponds to the percentage of pixels used for training and LEARNING\_RATE corresponds to the initial learning rate. All of them are defined in *lib/config.py*. Also EPOCH corresponds to the current checkpoint epoch according to PERIOD.

### Test

> ./test.py BO\_EPOCH IP\_EPOCH KSC\_EPOCH PU\_EPOCH SV\_EPOCH

Will perform the necessary tests to generate the *reliability diagram* and the *accuracy vs. uncertainty* plots, along with the *class uncertainty* plot of each image. For that, it is necessary that the five dataset models are already trained. The plots will be saved in *Test/*.

The five mandatory *epoch* parameters of *./test.py* correspond with the number of epochs of the selected checkpoint for testing each model in this order: BO, IP, KSC, PU and SV. In case you want to eliminate or add datasets, take on account that this order must correspond with the *DATASETS_LIST* variable in *lib/config.py*.

### Test map

Here we call the entire hyperspectral image a *map*, that is, every pixel on its original position to conform the image; and not only the labelled pixels, but the unlabelled too.

> ./test\_map.py NAME EPOCH

Will perform the inference of every pixel of the NAME dataset and generate a pdf image called *H_{NAME}.pdf* containing the RGB image, the ground truth, the prediction (with a different colour for each class) and the uncertainty map (with a different colour for each range of uncertainty). The images will be saved in *Test/*.

As in *test.py*, EPOCH refers to the number of epochs of the selected checkpoint.

### Test with noisy data

> ./test\_noise.py BO\_EPOCHS IP\_EPOCHS KSC\_EPOCHS PU\_EPOCHS SV\_EPOCHS

Will perform the necessary tests to generate the *combined noise* plot. For that, it is necessary that the five dataset models are already trained. The plots will be saved in *Test/*.

The parameters are the same of *test.py* and they behave the same way.

### Train with mixed classes

The exact same execution of *train.py* activating the *-m* flag will generate the trained model with mixed classes.

The resultant trained checkpoints will be in *Models/{NAME}\_{LAYER1\_NEURONS}-{LAYER2\_NEURONS}model\_{P\_TRAIN}train\_{LEARNING\_RATE}lr\_{CLASS\_A}-{CLASS\_B}mixed/* with an *epoch\_{EPOCH}* directory for each checkpoint.

Where CLASS\_A and CLASS\_B correspond to the numbers of the mixed classes, which are defined for each dataset in *lib/config.py*.

### Test models with mixed classes

> ./test\_mixed.py BO\_EPOCHS IP\_EPOCHS KSC\_EPOCHS PU\_EPOCHS SV\_EPOCHS

Will perform the necessary tests to generate and print a table with the *aleatoric uncertainty* of the mixed classes and the *mixed classes* plot of each model. For that, it is necessary that the five dataset models are already trained. The results will be saved in *Test/*.

The parameters are the same of *test.py* and they behave the same way.
