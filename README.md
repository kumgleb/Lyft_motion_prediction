# Lyft motion prediction

This repository contains pretrainded models for the motion prediction task based on Lyft Level 5 Prediction dataset. <br>
Complete model utilizes BEV-semantic map of the frame with agents histories and predict multi-modal future trajectories of agent for the next 5 seconds with the frequency of 10Hz. 
<br/><br/>
![](images/example.png)

## Examples:
Training and prediciton of models with examples are provided in the notebooks:
 * [Train example](https://github.com/kumgleb/Lyft_motion_prediction/blob/main/examples/train_example.ipynb)
 * [Prediction example](https://github.com/kumgleb/Lyft_motion_prediction/blob/main/examples/prediction_example.ipynb)

## Models:
The complete model consists of two stages:
1. CVAE-based model that learns a distribution P(trajectory | frame, history)
![](images/Arcitecture.png)
**Loss function:** <br>
![](images/loss_cvae.PNG) <br>
Also maximum mean discrepancy loss is supported and can be used to train the model.

2. The Extractor model is based on CVAE model and extract multi-modal prediction for the given future trajectories probabilities:
<br/><br/>
![](images/Extractor_model.png)
Ground truth is assumed to be a mixture of multi-demensional independent Normal distributions over time. <br>
**Loss function:** <br>
![](images/loss_extractor.PNG)

## Dataset:
Dataset consists of more then 1000 hours of data collected by a fleet of 20 autonomous vehicles along a fixed route in Palo Alto.
Data about self-driving vehicle and other traffic participants is represented in chunks of 25 seconds duration.
Also high-definition semantic map and high-resolution aerial picture are provided.

* Download dataset:
https://self-driving.lyft.com/level5/data/

* Paper about dataset:
https://arxiv.org/pdf/2006.14480.pdf

* Official development tools for the dataset:
https://github.com/lyft/l5kit/



