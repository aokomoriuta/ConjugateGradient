#pragma OPENCL EXTENSION cl_khr_fp64: enable

//! set vector's value
/*!
	\param vector target vector
	\param value value to set
*/
__kernel void SetAllVector(
	__global double* vector,
	const double value)
{
	// get element index
	int i = get_global_id(0);

	// set value
	vector[i] = value;
	
}

//! Add each vector element
/*!
	\param answer vector which answer is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void PlusEachVector(
	__global double* answer,
	__global const double* left,
	__global const double* right,
	const double C)
{
	// get element index
	int i = get_global_id(0);

	// add each element
	answer[i] = left[i] + C*right[i];
}

//! Multiply each vector element
/*!
	\param answer vector which answer is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void MultiplyEachVector(
	__global double* answer,
	__global const double* left,
	__global const double* right)
{
	// get element index
	int i = get_global_id(0);

	// multiply each element
	answer[i] = left[i] * right[i];

}

//! Add each values at a vector's second half to its first half
/*!
	\param count size of vector
	\param target vector
*/
__kernel void AddVectorSecondHalfToFirstHalf(
	const long count,
	__global double* values)
{
	// get number and index
	long globalIndex = get_global_id(0);
	long globalSize = get_global_size(0);

	// calculate latter index
	long nextIndex = globalIndex + globalSize;

	// only in vector's size
	if(nextIndex < count)
	{
		// add second half value to this
		values[globalIndex] += values[nextIndex];
	}
}

//! Multiply each matrix element by vector element
/*!
	\param answer which answer is stored to
	\param matrixElements elements of matrix
	\param matrixColumnIndeces column index of the matrix element which stored that position 
	\param matrixNonzeroCounts number of elements which is not zero
	\param vector right vector
*/
__kernel void MultiplyMatrixVector(
	__global double* answer/*,
	__global const double* matrixElements,
	__global const long* matrixColumnIndeces,
	__global const long* matrixNonzeroCounts,
	__global const double* vector*/)
{
	// get maximum number of elements
	long maxNonzeroCount = get_global_size(1);

	// get element index
	long i = get_global_id(0);
	long k = get_global_id(1);

	answer[i*maxNonzeroCount + k] = k;
	
	/*// if this is not zero value
	if(k < matrixNonzeroCounts[i])
	{
		// get global index in matrix
		long matrixIndex = i*maxNonzeroCount + k;

		// get columnIndex;
		long j = matrixColumnIndeces[matrixIndex];

		// multiply
		answer[matrixIndex] = matrixElements[matrixIndex] * vector[j];
	}*/
}
	