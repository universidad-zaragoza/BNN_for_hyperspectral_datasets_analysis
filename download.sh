#!/bin/bash

# Main directory
# -----------------------------------------------------------------------------
mkdir bnn4hi
cd bnn4hi

# Non executable files
for file in Dockerfile README.md __init__.py
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/$file
    chmod 664 $file
done

# Executable files
for file in launch.sh test.py test_map.py test_mixed.py test_noise.py train.py
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/$file
    chmod 775 $file
done

# lib directory
# -----------------------------------------------------------------------------
mkdir lib
cd lib

# Non executable files
for file in D_illuminants.mat HSI2RGB.py README.md __init__.py
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/lib/$file
    chmod 664 $file
done

# Executable files
for file in analysis.py config.py data.py model.py plot.py
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/lib/$file
    chmod 775 $file
done

# docs directory
# -----------------------------------------------------------------------------
cd ..
mkdir docs
cd docs

# Non executable files
for file in bnn4hi.html index.html search.js
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/docs/$file
    chmod 664 $file
done

# docs/bnn4hi directory
# -----------------------------------------------------------------------------
mkdir bnn4hi
cd bnn4hi

# Non executable files
for file in lib.html test.html test_map.html test_mixed.html test_noise.html train.html
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/docs/bnn4hi/$file
    chmod 664 $file
done

# docs/bnn4hi/lib directory
# -----------------------------------------------------------------------------
mkdir lib
cd lib

# Non executable files
for file in HSI2RGB.html analysis.html config.html data.html model.html plot.html
do
    wget -O $file https://github.com/universidad-zaragoza/BNN_for_hyperspectral_datasets_analysis/raw/main/docs/bnn4hi/lib/$file
    chmod 664 $file
done
