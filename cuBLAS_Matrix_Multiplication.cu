/*
* cuBLAS_Matrix_Multiplication.cu - MATLAB external interfaces GPU implementation of a matrix multiplication using cuBLAS library.
*
*
* Input:
*	A	- MxN Matrix in float-precision
*	B 	- NxP Matrix in float-precision
* Output:
*	C	- MxP Matrix in float-precision
*
*
*
* The calling syntax from Matlab is:
*
*	C = cuBLAS_Matrix_Multiplication(A,B);
*
* This is a MEX file for MATLAB
*
*
*/


#include "mex.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

/* Input error handling */
/*--------------------------------------------------------------------------------------------------*/
void inputErrorHandling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    /* Verify if the inputs have a valid shape and that all requred inputs are given.
     *
     * Expected input from matlab: (MxN real float, NxP real float)
     *
     * Expected output to matlab: MxP real float
     *
    */
  
	// Check if two inputs are given
    if(nrhs != 2)
      mexErrMsgIdAndTxt( "parallel:gpu:cuBLAS_Matrix_Multiplication:invalidNumInputs",
              "Two inputs required.");
			  
    // Check if not more than one output is requested
	if(nlhs > 1)
      mexErrMsgIdAndTxt( "parallel:gpu:cuBLAS_Matrix_Multiplication:maxlhs",
              "Only one output argument of size MxP allowed");
	
	// Check if the input float Array
	if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]) ) {
        mexErrMsgIdAndTxt("parallel:gpu:cuBLAS_Matrix_Multiplication:InvalidInput", "Input has to be float");
    }
    
    // Check if number of columns of the first input is equal to the number of rows of the second input
	if (mxGetN(prhs[0]) != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("parallel:gpu:cuBLAS_Matrix_Multiplication:InvalidInputSize", "column(A) != row(B)");
    }
	
}


/* Host code */
/*--------------------------------------------------------------------------------------------------*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){


	//	Variable declarations
	float *device_A, *device_B, *device_C;
	const float *A, *B;
	float *C;
	int num_A_rows, num_A_cols;
	int num_B_rows, num_B_cols;
	int num_C_rows, num_C_cols;


	// Input Error Handling
	inputErrorHandling(nlhs, plhs, nrhs, prhs);
	 
	
	// Read the inputs from Matlab
	A = (float *)mxGetData(prhs[0]);
	B = (float *)mxGetData(prhs[1]);


	// Get the dimension of the input array
	num_A_rows = (int)mxGetM(prhs[0]);
	num_A_cols = (int)mxGetN(prhs[0]);
	num_B_rows = (int)mxGetM(prhs[1]);
	num_B_cols = (int)mxGetN(prhs[1]);
	num_C_rows = num_A_rows;
	num_C_cols = num_B_cols;


	
	// Initialize the output to MATLAB
	plhs[0] = mxCreateNumericMatrix(num_C_rows, num_C_cols, mxSINGLE_CLASS, mxREAL);
	C = (float *)mxGetData(plhs[0]);
    

	// Allocate GPU memory
	cudaMalloc(&device_A, sizeof(float) * num_A_rows * num_A_cols);
	cudaMalloc(&device_B, sizeof(float) * num_B_rows * num_B_cols);
	cudaMalloc(&device_C, sizeof(float) * num_C_rows * num_C_cols);
   
	
	// Create handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	// Set the Matrices
	cublasSetMatrix(num_A_rows, num_A_cols, sizeof(float), A, num_A_rows, device_A, num_A_rows);
	cublasSetMatrix(num_B_rows, num_B_cols, sizeof(float), B, num_B_rows, device_B, num_B_rows);

	
	// Scaling facors for matrix Multiplication: C = (alpha*A)*b + (beta*c)
	float alpha = 1.0;
	float beta = 0.0;
	
	
	// Matrix multiplication using cuBLAS: (m X n) * (n X p) = (m X p)
	// Signature: handle, operation, operation, m, n, p, alpha, A, lda, B, ldb, beta, C, ldc
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_A_rows, num_B_cols, num_A_cols, &alpha, device_A, num_A_rows, device_B, num_B_rows, &beta, device_C, num_C_rows);
   
	// Copy C back to host
	cudaMemcpy(C,device_C, sizeof(float) * num_C_rows * num_C_cols, cudaMemcpyDeviceToHost);        
    

	// Clearing GPU memory cache

    cublasDestroy(handle);
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

}
