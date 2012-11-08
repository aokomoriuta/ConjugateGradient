#include"Vector_Int.hpp"
#include<thrust/copy.h>

extern "C"
{
	// �x�N�g���̍쐬
	__declspec(dllexport) VectorInt* _stdcall Create_Int(const int size, const int deviceID)
	{
		::cudaSetDevice(deviceID);

		VectorInt* vec = new VectorInt(size);

		return vec;
	}

	// �z��֕���
	__declspec(dllexport) void _stdcall CopyTo_Int(const VectorInt* source, int destination[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source->begin(), count, destination);
	}

	// �z�񂩂畡��
	__declspec(dllexport) void _stdcall CopyFrom_Int(VectorInt* destination, const int source[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source, count, destination->begin());
	}
	
	// �x�N�g���̔p��
	__declspec(dllexport) void _stdcall Delete_Int(VectorInt* vec, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		delete vec;
	}
}