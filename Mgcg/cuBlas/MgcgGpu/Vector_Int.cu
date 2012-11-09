#include"Vector_Int.hpp"
#include<thrust/copy.h>
#include<thrust/transform.h>


extern "C"
{
	// �x�N�g���̍쐬
	__declspec(dllexport) VectorInt* _stdcall Create_Int(const int size)
	{
		VectorInt* vec = new VectorInt(size);

		return vec;
	}

	// �z��֕���
	__declspec(dllexport) void _stdcall CopyToArray_Int(const VectorInt* source, int destination[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source->begin() + sourceOffset, count, destination + destinationOffset);
	}

	// �z�񂩂畡��
	__declspec(dllexport) void _stdcall CopyFromArray_Int(VectorInt* destination, int source[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source + sourceOffset, count, destination->begin() + destinationOffset);
	}
	
	// �x�N�g���̔p��
	__declspec(dllexport) void _stdcall Delete_Int(VectorInt* vec)
	{
		delete vec;
	}
	
	// CUDA�p�|�C���^�֕ϊ�����
	__declspec(dllexport) int* _stdcall ToRawPtr_Int(VectorInt* vec)
	{
		return thrust::raw_pointer_cast(&(*vec)[0]);
	}
}