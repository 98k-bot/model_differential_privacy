# ICLR-2020

## Introduction

This repository contains the original implementation of the paper - **Model more Intelligent than I want it to be? A Framework for Measuring and Improving Deep Learning Model Trust**.


## Quick Start


### Install Prerequisites

```
$ git clone https://github.com/dl-model-recommend/model-trust.git

$ cd model-trust

$ pip install -r requirements.txt
```

### Create synthetic dataset

```
$ python make_dataset.py -n 5
```
here: n is num of image to generate per variation. 


### Train densenet model on sysnthetic data 
```
$ python train_models.py --task color --prefix densenet --epoch 50
```
here: task is `color, shape, size, quadrant, background`
prefix is unique identifier for experiments, like : densent


### To extract the features from trained model
```
$ python extract_features.py --task color --prefix densenet --layer flatten_1
```


### To calculate accuracy on all auxillary task 
```
$ python classification_net.py --task color --prefix densenet --epoch 50
```

### Calculate model trust

```
$ python caluculate_mode_trust.py 
```