#include"Vector_Double.hpp"
#include<thrust/copy.h>

extern "C"
{
	// �x�N�g���̍쐬
	__declspec(dllexport) Vector* _stdcall Create_Double(const int size, const int deviceID)
	{
		::cudaSetDevice(deviceID);

		Vector* vec = new Vector(size);

		return vec;
	}

	// �z��֕���
	__declspec(dllexport) void _stdcall CopyTo_Double(const Vector* source, double destination[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source->begin(), count, destination);
	}

	// �z�񂩂畡��
	__declspec(dllexport) void _stdcall CopyFrom_Double(Vector* destination, const double source[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source, count, destination->begin());
	}
	
	// �x�N�g���̔p��
	__declspec(dllexport) void _stdcall Delete_Double(Vector* vec, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		delete vec;
	}
}