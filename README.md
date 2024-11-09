# DHHNN
DHHNN: A Dynamic Hypergraph Hyperbolic Neural Network Based on Variational Autoencoder for Multimodal Data Integration and Node Classification


## The framework of DHHNN
![image]((https://github.com/HaoWuLab-Bioinformatics/DHHNN/blob/main/model.png))

## Overview
The folder "**data**" contains all data used.  
The folder "**dhg**" contains code for building hypergraphs.  
The folder "**layers**" contains code for each layer.  
The folder "**manifolds**" contains implementation code for hyperbolic space,Euclidean space and poincare space.  
The folder "**models**" contains the implementation code of the VAE framework.  
The folder "**optimizers**" contains optimizations to the Riemannian Adam algorithm.  
The folder "**src**" contains code for parsing command line arguments and providing configuration information for the program.  
The folder "**utils**" contains function definition, data set import and other codes.  

## Dependency
Mainly used libraries:  
Python 3.9.15  
numpy==1.16.2  
scikit-learn==0.20.3  
torch==1.1.0  
torchvision==0.2.2  
networkx==2.2  

See "**requirements.txt"** for all detailed libraries.  
Other developers can use the following command to install the dependencies contained in "**requirements.txt"**:
`pip install -r requirements.txt`  
