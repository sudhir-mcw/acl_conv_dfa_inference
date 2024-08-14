import torch
from torch import nn
import os
import numpy as np

IN_CHANNELS = 3
OUT_CHANNELS = 64
KERNEL_SIZE = 3

if __name__ == "__main__":
    # load input, weight
    input_tensor_1 = torch.from_numpy(np.load("new_test/input/cpp_input_1.npy"))
    print("input shape ", input_tensor_1.shape)
    weight_tensor = torch.ones(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)
    print("weight shape ", weight_tensor.shape)
    weight_tensor = torch.from_numpy(np.load("new_test/weight/cpp_weight.npy"))
    weight_tensor = weight_tensor.squeeze()
    print("weight shape ", weight_tensor.shape)

    # perform convolution
    conv = nn.Conv2d(
        OUT_CHANNELS, 3, kernel_size=KERNEL_SIZE, padding=0, bias=False, stride=1
    )
    conv.weight.data = weight_tensor
    output_tensor_1 = conv(input_tensor_1)
    print("output_tensor_1 shape ", output_tensor_1.shape)
    np.save("new_test/output/py_output_wtpad_1.npy", output_tensor_1.detach().numpy())

    # load input and use the same weigths
    input_tensor_2 = torch.from_numpy(np.load("new_test/input/cpp_input_2.npy"))
    print("input shape ", input_tensor_2.shape)
    output_tensor_2 = conv(input_tensor_2)
    print("output_tensor_2 shape ", output_tensor_2.shape)
    np.save("new_test/output/py_output_wtpad_2.npy", output_tensor_2.detach().numpy())

    # merge the two outputs into one tensor
    merged_output = torch.cat((output_tensor_1, output_tensor_2), dim=2)
    print("merged_output shape ", merged_output.shape)
    np.save(
        "new_test/output/py_output_wtpad_merged.npy", merged_output.detach().numpy()
    )
