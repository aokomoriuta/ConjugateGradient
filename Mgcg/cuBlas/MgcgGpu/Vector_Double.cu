#include"Vector_Double.hpp"
#include<thrust/copy.h>

extern "C"
{
	// �x�N�g���̍쐬
	__declspec(dllexport) Vector* _stdcall Create_Double(const int size)
	{
		Vector* vec = new Vector(size);

		return vec;
	}

	// �z��֕���
	__declspec(dllexport) void _stdcall CopyToArray_Double(const Vector* source, double destination[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source->begin() + sourceOffset, count, destination + destinationOffset);
	}

	// �z�񂩂畡��
	__declspec(dllexport) void _stdcall CopyFromArray_Double(Vector* destination, const double source[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source + sourceOffset, count, destination->begin() + destinationOffset);
	}
	
	// �x�N�g���̔p��
	__declspec(dllexport) void _stdcall Delete_Double(Vector* vec)
	{
		delete vec;
	}
	
	// CUDA�p�|�C���^�֕ϊ�����
	__declspec(dllexport) double* _stdcall ToRawPtr_Double(Vector* vec)
	{
		return thrust::raw_pointer_cast(&(*vec)[0]);
	}

	// �f�o�C�X�ԂŃf�[�^��]������
	__declspec(dllexport) void _stdcall CopyFromDevice_Double(const double* source, double* destination,
		const int count, const int sourceOffset, const int destinationOffset)
	{
		::cudaMemcpy(destination + destinationOffset, source + sourceOffset, count, cudaMemcpyDeviceToDevice);
	}
}