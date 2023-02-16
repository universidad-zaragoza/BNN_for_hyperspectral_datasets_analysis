# lib

This library contains the internal modules of bnn4hi package.

## modules

### config.py

*Config module of the BNN4HI package*

This module provides macros with data information, training and testing
parameters and plotting configurations.

### data.py

*Data module of the BNN4HI package*

The functions of this module are used to load, preprocess and organise
data from hyperspectral datasets.

### model.py

*Model module of the BNN4HI package*

This module defines the bayesian model used to train.

### analysis.py

*Analysis module of the BNN4HI package*

The functions of this module are used to generate and analyse bayesian
predictions.

### plot.py

*Plot module of the BNN4HI package*

The functions of this module are used to generate plots using the
results of the analysed bayesian predictions.

## HSI2RGB

*Method to create quality RGB images from hyperspectral images*

Files HSI2RGB.py and D_illuminants.mat are used in `plot.py` to generate RGB images from hyperspectral images. Both files are extracted from their original repository:

[https://github.com/JakobSig/HSI2RGB](https://github.com/JakobSig/HSI2RGB)

This method is published in:

M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson, H. Deborah and J. R. Sveinsson, "Creating RGB Images from Hyperspectral Images Using a Color Matching Function," IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium, 2020, pp. 2045-2048, doi: 10.1109/IGARSS39084.2020.9323397.
