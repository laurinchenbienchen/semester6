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

print('the defalt device is:',torch.get_default_device())
print('the object is a tensor:',torch.is_tensor(tensor))
print('the input is a complex datatype:',torch.is_complex(tensor))
print()

# standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1)
print(y2)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor)
print(z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)

# **Single-element tensors**
# If you have a one-element tensor, for example by aggregating all
# values of a tensor into one value, you can convert it to a Python
# numerical value using ``item()``:

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# tensor to numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# change in the tensor reflects in numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}")

# changes in the numpy array reflect in the tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
print("Using torch", torch.__version__)
print(f"Device tensor is stored on: {tensor.device}")