#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

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

//! Multiply each matrix element by vector element
/*!
	\param answer which answer is stored to
	\param matrixElements elements of matrix
	\param matrixColumnIndeces column index of the matrix element which stored that position 
	\param matrixNonzeroCounts number of elements which is not zero
	\param vector right vector
*/
__kernel void MultiplyMatrixVector(
	__global double* answer,
	__global const double* matrixElements,
	__global const long* matrixColumnIndeces,
	__global const long* matrixNonzeroCounts,
	__global const double* vector)
{
	// get maximum number of elements
	long maxNonzeroCount = get_global_size(1);

	// get element index
	long i = get_global_id(0);
	long k = get_global_id(1);
	
	// get global index in matrix
	long matrixIndex = i*maxNonzeroCount + k;

	// if this is not zero value
	if(k < matrixNonzeroCounts[i])
	{
		// get columnIndex;
		long j = matrixColumnIndeces[matrixIndex];

		// multiply
		answer[matrixIndex] = matrixElements[matrixIndex] * vector[j];
	}
	// otherwise
	else
	{
		// zero
		answer[matrixIndex] = 0;
	}
}

//! column vector to row vector
/*!
	\param rowVector transoised vector
	\param columnVector transposing vector
	\param rowCount size of row of columnVector
*/
__kernel void ColumnVectorToRow(
	__global double* rowVector,
	__global const double* columnVector,
	const long rowCount)
{
	// get number and index
	long i = get_global_id(0);

	// store element
	rowVector[i] = columnVector[i*rowCount];
}

//! Add each values on second half of a row to its first half
/*!
	\param count target size of column
	\param maxCount maximum size of one column
	\param target vector
*/
__kernel void AddEachLocalValuesToTop(
	const long count,
	const long maxCount,
	__global double* values,
	__local double* localValues)
{
	// get number and index
	const long globalIndexI = get_global_id(0);
	const long localIndexJ = get_local_id(1);
	const long localSizeJ = get_local_size(1);
	const long groupIndexJ = get_group_id(1);
	const long groupSizeJ = get_num_groups(1);
	
	// calculate offset by row number
	const long rowOffset = globalIndexI*maxCount;

	// calculate local and global total index
	const long localIndex1 = 2*localIndexJ;
	const long localIndex2 = localIndex1+1;
	const long globalIndex1 = rowOffset + groupIndexJ + localIndex1 + ((localIndexJ == 0) ? 0 : groupSizeJ-1 + groupIndexJ*(localSizeJ*2-2));
	const long globalIndex2 = rowOffset + groupIndexJ + localIndex2 +                           groupSizeJ-1 + groupIndexJ*(localSizeJ*2-2);

	// copy values to local from grobal
	localValues[localIndex1] = (globalIndex1 - rowOffset < count) ? values[globalIndex1] : 0;
	localValues[localIndex2] = (globalIndex2 - rowOffset < count) ? values[globalIndex2] : 0;

	// synchronize work items in a group
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// while reduction
	for(long thisSize = localSizeJ; thisSize >= 1; thisSize/=2)
	{
		// only in target region for reduction
		if(localIndexJ < thisSize)
		{
			//printf("[%d] + [%d]\n", localIndex, localIndex + thisSize);

			// add second half value to first half
			localValues[localIndexJ] += localValues[localIndexJ + thisSize];
		}
		
		// synchronize work items in a group
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// if top in local
	if(localIndexJ == 0)
	{
		// store result to global
		values[rowOffset + groupIndexJ] = localValues[0];
	}
}