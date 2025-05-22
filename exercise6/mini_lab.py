import torch
import numpy as np

# torch version check
print("Using torch", torch.__version__)
# Using torch 2.7.0

# innitializing tensors

# tensors created directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

# tensorrs created from numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# from another tensor
x_ones = torch.ones_like(x_data)    #retains properties of x_data
print(f"ones tensor: \n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides datatype
print(f"random tensor: \n{x_rand}\n")

# with random or constant values
# shape: tupel of tensor dimensions, determines the dimensionality of the output tensor

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# attributes of a tensor
# tensor attributes describe shape, datatype, device on which they are stored

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# operations on tensors
# Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing,
# indexing, slicing), sampling

# By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using
# ``.to`` method (after checking for accelerator availability). Keep in mind that copying large tensors
# across devices can be expensive in terms of time and memory!

# moving tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print()
    print("Using torch", torch.__version__)
    print(f"Device tensor is stored on: {tensor.device}")