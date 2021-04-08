import torch
x = torch.rand(5, 3)
print(x)

print(f'Is cuda available in the system: {torch.cuda.is_available()}')

# Check that this time the variable is in the CPU
print(x.device)

# Now we change it to CUDA
x_cuda = x.to("cuda")
print(x_cuda.device)

# Show the size of the tensor

print(f'x.shape {x.shape}')
print(f'x.size {x.size()}')

# Convert the tensor to numpy array:
x_numpy = x_cuda.cpu().detach().numpy()

print(f'Numpy array: \n {x_numpy}')
