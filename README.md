# PPCon 1.0: Biogeochemical Argo Profile Prediction with 1D Convolutional Networks

Python implementation for paper "PPCon 1.0: Biogeochemical Argo Profile Prediction with 1D Convolutional Networks": 
Gloria Pietropolli, Luca Manzoni, and Gianpiero Cossarini

## Abstract
Effective observation of the ocean is vital for studying and assessing the state and evolution of the marine ecosystem, and for evaluating the impact of human activities. 
However, obtaining comprehensive oceanic measurements across temporal and spatial scales and for different biogeochemical variables remains challenging. 
Autonomous oceanographic instruments, such as Biogeochemical (BCG) Argo profiling floats, have helped expand our ability to obtain subsurface and deep ocean measurements, but measuring biogeochemical variables such as nutrient concentration still remains more demanding and expensive than measuring physical variables. 
Therefore, developing methods to estimate marine biogeochemical variables from high-frequency measurements becomes necessary. 
Current Neural Network (NN) models developed for this task are based on a Multilayer Perceptron (MLP) architecture, trained over punctual pairs of input-output features.
Thus, MLPs lack awareness of the typical shape of biogeochemical variable profiles they aim to infer, resulting in irregularities such as jumps and gaps when used for the prediction of vertical profiles.
In this study, we evaluate the effectiveness of a one-dimensional Convolutional Neural Network (1D CNN) model for predicting nutrient profiles, leveraging the typical shape of vertical profiles of a variable as a prior constraint during training. 
We will present a novel model named PPCon (Predict Profiles Convolutional), which is trained over a dataset containing BCG Argo float measurements, for the prediction of nitrate, chlorophyll, and Backscattering (bbp700) starting from date, geolocation, temperature, salinity, and oxygen.
The effectiveness of the model is then accurately validated by presenting both quantitative metrics and visual representations of the predicted profiles. 
Our proposed approach proves capable of overcoming the limitations of MLPs, resulting in smooth and accurate profile predictions.

## Instructions

Code runs with python 3.8.5 on Ubuntu 20.04, after installing the following requirements:  

```bash
pip install -r requirements.txt 
```

To run the code, enter the following command:

```bash
python3 main.py --variable --epochs --lr --dropout_rate --snaperiod --lambda_l2_reg --batch_size --alpha_smooth_reg
```
where the inputs arguments stand for: 
* `--variable` is the biogeochemical variable considered (that can be: _NITRATE_, _CHLA_, _BBP700_)  
* `--epochs` is the number of epochs for the training
*  `--lr` is the learning rate for the training
*  `--dropout_rate` is the dropout rate for the training
*  `--snaperiod` is the number of epochs after which the intermediate model is saved
*  `--lambda_l2_reg` set the multiplicative loss factor for the lambda regularization
*  `--batch_size` is the batch size for the training
*  `--alpha_smooth_reg` set the multiplicative loss factor for the smooth regularization

The dataset are already generated in a tensor form ready for the training, and splitted into train and test. 
The dataset are contained in the _ds_ folder. 

The codes that reproduce the plots of the paper are contained in the folder `analysis`:
* To get __Figure 2-4__ the functions are contained in _analysis/comparison_architecture.py_
* To get __Figure 5__ the functions are contained in _analysis/scatter_error.py_
* To get __Figure 6-8__ the functions are contained in _analysis/hovmoller_diagram.py_
* To get __Figure B1__ the functions are contained in _analysis/profile.py_
