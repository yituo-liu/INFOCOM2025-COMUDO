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

During the training of the non-convex neural network, the variables representing the channel state information (CSI) at all time slots become excessively large. Due to GPU memory limitations, it is not feasible to store and compute all the CSI values directly on the GPU.  

To address this issue, we have implemented a fixed random seed mechanism. This ensures that the CSI generated at each time slot remains consistent across individual algorithm code file. By doing so, the same CSI values are reproduced whenever the code is executed.