# ARM Compute Library Convolution Inference With Dataflow Architecture 
The repo contains Independent C++ inference with ARM compute Library for Convolution layer implementation with Dataflow Architecture

## Machine Requirements:
- Processor Architecture: ARM64
- RAM: Minimum 8GB
- OS: Ubuntu 20.04 
- Storage: Minimum 64GB

# Prequisites
* G++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
* cmake version 3.29.3
* GNU Make 4.2.1
* [cnpy](https://github.com/rogersce/cnpy) 
* [ARM Compute Library](https://github.com/arM-software/ComputeLibrary/)
* Python 3.8.10 

# Install Prequisites
1. Build the ARM Compute library by referring to the [documentation](https://arm-software.github.io/ComputeLibrary/latest/how_to_build.xhtml)
2. Build the cnpy library by following the steps in [documentation](https://github.com/rogersce/cnpy?tab=readme-ov-file#installation)  

# Cloning the Repo 
Clone the repo using the following command  
```
    git clone https://github.com/sudhir-mcw/acl_conv_dfa_inference.git
    cd acl_conv_dfa_inference
```  
Update CMake Configuration on successful prequisite installation
Open the CMakeLists.txt file in the root of the project directory and update the following
* ARM_COMPUTE_DIR  - Replace the <path_to_ARM_Compute_Library_Directory> with path to ARM Compute Directory located
* ARM_COMPUTE_LIBRARY -  Replace the <path_to_ARM_Compute_Library> with path to ARM Compute Library so located usually  /path_to_ARM_Compute_Library_Directory/build/libarm_compute.so
* CNPY_LIBRARY     - Replace the <path_to_CNPY_Library> with the built CNPY library so file's path  

# How to Run C++ Convolution Layer Inference
1. Build the C++ Convolution Layer Inference 
```
    cmake -B build -S .
    make -C build 
``` 
2. Run the program 
```
    ./build/conv_dfa_wt_pad
```    
# How to Run Python Convolution Layer Inference (For Verification Of C++ Output)
1. Install Required python packages
```
    pip install -r requirements.txt
```
2. Run the python script to get convolution layer output and dump it to output file
```
    python conv_dfa_wt_pad.py
```

# Compare the Output 
1. All the output files are stored in new_test/output/ folder, Comparison of files can be done using the compare.py file 
```
    python compare.py <file_1.npy> <file_2.npy>
```
Sample usage
```
    python compare.py new_test/output/cpp_output_wtpad_merged.npy new_test/output/py_output_wtpad_merged.npy
```
Sample output 
```
    $ python compare.py new_test/output/cpp_output_wtpad_merged.npy new_test/output/py_output_wtpad_merged.npy
    Files are identical upto 4 decimals
```
