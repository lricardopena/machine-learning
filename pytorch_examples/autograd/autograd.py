import torch


A = torch.tensor([10.], requires_grad=True)
B = torch.tensor([5.], requires_grad=True)

y = A**5 - B**4

print(y)

# dy/dA = 5*A**4 = 5(10)^4 = 50000
# dy/dB = -4*B**3 = -4(5)^3 = -500

# If we print the grad at this moment it will be None because we have not been call the function backward on y

print(f'A.grad:{A.grad} \nB.grad:{B.grad}')

y.backward()

print(f'A.grad:{A.grad} \nB.grad:{B.grad}')
