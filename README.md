# Code Requirements

This code requires the following libraries to run:  
- `copy`  
- `torch`  
- `torchvision`  
- `numpy`  
- `os`  
- `matplotlib`  
- `random`  

# Notes on Implementation
### Convex Logistic Regression
In the `main` file, different algorithms are called and images are outputted. All algorithm outputs can be completed in one run.

### Non-Convex Neural Network
For the non-convex neural network, please run different algorithm files separately. After all files have been run once and results have been saved, create an additional file to plot the images. 

The reason is as described below:

During the training of the non-convex neural network, the variables representing the channel state information (CSI) for all time slots are excessively large. Due to GPU memory limitations, it is not feasible to store and compute all the CSI values directly on the GPU.  

To address this issue, we implement a fixed random seed. This ensures that the CSI generated at each time slot remains consistent across individual algorithm code file. Therefore the CSI variables just store the value on one time slot.