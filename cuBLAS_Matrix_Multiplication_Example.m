% 
% This example demonstrates how to call the cuBLAS matrix multiplication
% via the C++ MEX API
%
% author: Patrick Stähli
% last update: 29.12.2020


% Set the MV_NVCC_PATH
%setenv('MW_NVCC_PATH','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin')

% Compile
% !  The usage of the cuBLAS library has to be specified via -lcublas.
% !  Further, the directory to the header files for CUDA and cuBLAS have to
% !  be specified via -I'PATH_TO_CUDA\v10.0\include
%mexcuda cuBLAS_Matrix_Multiplication.cu -lcublas -I'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include'

clear
% Run the multiplication
A = single(rand(2048));
B = single(rand(2048));
C = single(zeros(2048));

fprintf('Matrix Multiplication on the CPU \n');
tic
C = A*B;
toc


fprintf('cuBLAS based matrix Multiplication on the GPU \n');
gpuDev1 = gpuDevice(); 
tic; 
C = cuBLAS_Matrix_Multiplication(A, B); 
wait(gpuDev1); 
toc


%exit

