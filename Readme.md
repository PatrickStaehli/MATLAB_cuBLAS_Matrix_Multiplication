# MATLAB example of a GPU matrix multiplication using cuBLAS via the C++ MEX API

## Description
This is a basic example of how to use the CUDA-accelerated library cuBLAS to perform a matrix multiplication.
cuBLAS is accessed from MATLAB via the [C++ MEX API](https://ch.mathworks.com/help/matlab/cpp-mex-file-applications.html). 

## Requrements

### Window 10
This implementation was tested with Matlab R2020b under Windows 10. To successfully compile the .cu source file, Visual Studio 2017 and CUDA V10.0 need to be installed. 
The path of MW_NVCC_PATH can be set by calling setenv('MW_NVCC_PATH','PATH_TO_CUDA\v10.0\bin') in Matlab.

To use the cuBLAS library, it has to be specified when compiling the source file using the flag -lcublas. Furthermore, the directory to the header files for CUDA and cuBLAS have to
be specified via -I'PATH_TO_CUDA\v10.0\include.

mexcuda cuBLAS_Matrix_Multiplication.cu -lcublas -I'PATH_TO_CUDA\v10.0\include'
