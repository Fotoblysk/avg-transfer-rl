
import torch
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """
    Compute the softmax of vector x.

    Parameters:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Softmax of the input tensor.
    """
    return torch.nn.functional.softmax(x, dim=0)

# Generate t values from 0.001 to 10
t_values = np.linspace(1e-6, 2, 1000)

# Initialize lists to store softmax values
softmax_10t = []
softmax_t = []

# Compute softmax for each t
for t in t_values:
    x = torch.tensor([2 * 1/t, 1/t], dtype=torch.float32)
    softmax_vals = softmax(x)
    softmax_10t.append(softmax_vals[0].item())
    softmax_t.append(softmax_vals[1].item())

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, softmax_10t, label='softmax(10t)')
plt.plot(t_values, softmax_t, label='softmax(t)')
plt.title('Softmax of [10t, t] for t in [0.001, 10]')
plt.xlabel('t')
plt.ylabel('Softmax value')
plt.legend()
plt.grid(True)
plt.show()