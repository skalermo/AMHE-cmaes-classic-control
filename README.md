# AMHE-cmaes-classic-control

This repository contains implementation of algorithm which optimizes parameters
of neural network using Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
for solving [classic control problems](https://gym.openai.com/envs/#classic_control).

## Requirements

- Python 3.8

## Installation

In order to run the project make sure to install the dependencies:

```bash
pip install -r requirements.txt
```

## Testing

To run the tests simply:

```bash
python -m unittest discover test
```

## Try it out!

  You can run simple script prepared for demonstration purposes. In [models/](models/) there are 
  some pretrained models so you don't have to wait ;)

```bash
python example.py
```
