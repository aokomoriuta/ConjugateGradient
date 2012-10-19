#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

//! REAL is provided by compiler option
typedef REAL Real;

//! Add each element
/*!
	\param result vector which result is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void AddVectorVector(
	__global Real* result,
	const __global Real* left,
	const __global Real* right,
	const Real C)
{
	// get element index
	int i = get_global_id(0);
	
	// add each element
	result[i] = left[i] + C * right[i];
}


//! Multiply one element per one work-item
/*!
	\param result vector which result is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void MultiplyVectorVector(
	__global Real* result,
	const __global Real* left,
	const __global Real* right)
{
	// get element index
	const int i = get_global_id(0);

	// multiply each element
	result[i] = left[i] * right[i];
}

//! Sum array by reduction
/*!
	\param values target array
	\param count number of elements
	\param nextOffset offset of next element of this element
*/
__kernel void ReductionSum(
	__global Real* values,
	const int count,
	__local Real* localValues)
{
	// get local size
	const int localSize = get_local_size(0);

	// get group index and size
	const int groupID = get_group_id(0);
	const int groupSize = get_num_groups(0);

	// get this element's index
	const int iLocal = get_local_id(0);
	const int iGlobal1 = ( (iLocal == 0) ? groupID : (groupSize + groupID*(localSize-1) + iLocal - 1) )<<1;
	const int iGlobal2 = iGlobal1 + 1;

	// copy values from 2 global values to local
	localValues[iLocal] = 
	(iGlobal1 < count) ?
		values[iGlobal1] + ((iGlobal2 < count) ? values[iGlobal2] : 0)
		: 0;

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// for each half
	for(int halfSize = localSize>>1; halfSize >= 1; halfSize >>= 1)
	{
		// get second half element's index
		const int jLocal = iLocal + halfSize;

		// only in first half
		if(iLocal < halfSize)
		{
			// add second half values to this
			localValues[iLocal] += localValues[jLocal];
		}

		// synchronize work items in this group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// only the first work-item in this group
	if(iLocal == 0)
	{
		// store sum of this group to global value
		values[groupID] = localValues[0];
	}
}

//! Seach max absolute value in array by reduction
/*!
	\param values target array
	\param count number of elements
	\param localValues local buffer
*/
__kernel void ReductionMaxAbsolute(
	__global Real* values,
	const int count,
	__local Real* localValues)
{
	// get local size
	const int localSize = get_local_size(0);

	// get group index and size
	const int groupID = get_group_id(0);
	const int groupSize = get_num_groups(0);

	// get this element's index
	const int iLocal = get_local_id(0);
	const int iGlobal1 = ( (iLocal == 0) ? groupID : (groupSize + groupID*(localSize-1) + iLocal - 1) )<<1;
	const int iGlobal2 = iGlobal1 + 1;

	// copy values from 2 global values to local
	localValues[iLocal] = 
	(iGlobal1 < count) ?
		max( fabs(values[iGlobal1]), ((iGlobal2 < count) ? fabs(values[iGlobal2]) : 0) )
		: 0;

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// for each half
	for(int halfSize = localSize>>1; halfSize >= 1; halfSize >>= 1)
	{
		// get second half element's index
		const int jLocal = iLocal + halfSize;

		// only in first half
		if(iLocal < halfSize)
		{
			// store larger value
			localValues[iLocal] = max(localValues[iLocal], localValues[jLocal]);
		}

		// synchronize work items in this group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// only the first work-item in this group
	if(iLocal == 0)
	{
		// store sum of this group to global value
		values[groupID] = localValues[0];
	}
}


//! Multiply matrix by vector
/*!
	\param count length of vector elements
	\param result vector which result is stored to
	\param matrix multiplied vector
	\param vector multiplying vector
	\param columnIndeces indeces of column of matrix
	\param nonzeroCount count of matrix element which is not zero per row
*/
__kernel void Matrix_x_Vector(
	const int count,
	__global Real* result,
	const __global Real* matrix,
	const __global Real* vector,
	const __global int* columnIndeces,
	const __global int* nonzeroCount,
	const int bufferSize,
	__local Real* localVector)
{
	// get element index
	const int globalID = get_global_id(0);

	// get other index and size
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupID = get_group_id(0);
	const int vectorFirstID = max(0, localSize *  groupID - bufferSize);
	const int vectorLastID  = min(count, localSize * (groupID + 1) + bufferSize); 

	// for each local vector
	for(int i = localID; i < vectorLastID  - vectorFirstID; i += localSize)
	{
		// copy from global
		localVector[i] = vector[vectorFirstID + i];
	}

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// ignore if index is larger than row count
	if(globalID > count) return;

	// initialize result
	Real thisResult = 0;

	// for all non-zero row
	for(int j = 0; j < nonzeroCount[globalID]; j++)
	{
		// add matrix multiplied by vector
		thisResult += matrix[globalID * MAX_NONZERO_COUNT + j] * localVector[columnIndeces[globalID * MAX_NONZERO_COUNT + j] - vectorFirstID];
	}

	// store result
	result[globalID] = thisResult;
}