import torch
x = torch.rand(5, 3)
print(x)

print(f'Is cuda available in the system: {torch.cuda.is_available()}')
