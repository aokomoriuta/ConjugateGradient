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
	__global const int* matrixColumnIndeces,
	__global const int* matrixNonzeroCounts,
	__global const double* vector)
{
	// get maximum number of elements
	int maxNonzeroCount = get_global_size(1);

	// get element index
	int i = get_global_id(0);
	int k = get_global_id(1);
	
	// get global index in matrix
	int matrixIndex = i*maxNonzeroCount + k;

	// if this is not zero value
	if(k < matrixNonzeroCounts[i])
	{
		// get columnIndex;
		int j = matrixColumnIndeces[matrixIndex];

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
	const int rowCount)
{
	// get number and index
	int i = get_global_id(0);

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
	const int count,
	const int maxCount,
	__global double* values,
	__local double* localValues)
{
	// get number and index
	const int globalIndexI = get_global_id(0);
	const int localIndexJ = get_local_id(1);
	const int localSizeJ = get_local_size(1);
	const int groupIndexJ = get_group_id(1);
	const int groupSizeJ = get_num_groups(1);
	
	// calculate offset by row number
	const int rowOffset = globalIndexI*maxCount;

	// calculate local and global total index
	const int localIndex1 = 2*localIndexJ;
	const int localIndex2 = localIndex1+1;
	const int globalIndex1 = rowOffset + groupIndexJ + localIndex1 + ((localIndexJ == 0) ? 0 : groupSizeJ-1 + groupIndexJ*(localSizeJ*2-2));
	const int globalIndex2 = rowOffset + groupIndexJ + localIndex2 +                           groupSizeJ-1 + groupIndexJ*(localSizeJ*2-2);

	// copy values to local from grobal
	localValues[localIndex1] = (globalIndex1 - rowOffset < count) ? values[globalIndex1] : 0;
	localValues[localIndex2] = (globalIndex2 - rowOffset < count) ? values[globalIndex2] : 0;

	// synchronize work items in a group
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// while reduction
	for(int thisSize = localSizeJ; thisSize >= 1; thisSize/=2)
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

//! Add each values on second half of a row to its first half
/*!
	\param count target size of column
	\param maxCount maximum size of one column
	\param target vector
*/
__kernel void SumEachColumnValues(
	const int count,
	__global double* values)
{
	// get number and index
	const int i = get_global_id(0);

	const int rowOffset = i*count;

	for(int j = 1; j < count; j++)
	{
		values[rowOffset] += values[rowOffset+j];
	}
}



//! Add each values on second half of a row to its first half
/*!
	\param count target size of column
	\param maxCount maximum size of one column
	\param target vector
*/
__kernel void StoreMaxEachLocalValuesToTop(
	const int count,
	const int maxCount,
	__global double* values,
	__local double* localValues)
{
	// get number and index
	const int globalIndexI = get_global_id(0);
	const int localIndexJ = get_local_id(1);
	const int localSizeJ = get_local_size(1);
	const int groupIndexJ = get_group_id(1);
	const int groupSizeJ = get_num_groups(1);
	
	// calculate offset by row number
	const int rowOffset = globalIndexI*maxCount;

	// calculate local and global total index
	const int localIndex1 = 2*localIndexJ;
	const int localIndex2 = localIndex1+1;
	const int globalIndex1 = rowOffset + groupIndexJ + localIndex1 + ((localIndexJ == 0) ? 0 : groupSizeJ-1 + groupIndexJ*(localSizeJ*2-2));
	const int globalIndex2 = rowOffset + groupIndexJ + localIndex2 +                           groupSizeJ-1 + groupIndexJ*(localSizeJ*2-2);

	// copy values to local from grobal
	localValues[localIndex1] = (globalIndex1 - rowOffset < count) ? fabs(values[globalIndex1]) : 0;
	localValues[localIndex2] = (globalIndex2 - rowOffset < count) ? fabs(values[globalIndex2]) : 0;

	// synchronize work items in a group
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// while reduction
	for(int thisSize = localSizeJ; thisSize >= 1; thisSize/=2)
	{
		// only in target region for reduction
		if(localIndexJ < thisSize)
		{
			// add second half value to first half
			localValues[localIndexJ] = max(localValues[localIndexJ], localValues[localIndexJ + thisSize]);
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