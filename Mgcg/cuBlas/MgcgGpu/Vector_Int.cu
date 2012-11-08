#include"Vector_Int.hpp"
#include<thrust/copy.h>

extern "C"
{
	// ベクトルの作成
	__declspec(dllexport) VectorInt* _stdcall Create_Int(const int size, const int deviceID)
	{
		::cudaSetDevice(deviceID);

		VectorInt* vec = new VectorInt(size);

		return vec;
	}

	// 配列へ複製
	__declspec(dllexport) void _stdcall CopyTo_Int(const VectorInt* source, int destination[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source->begin(), count, destination);
	}

	// 配列から複製
	__declspec(dllexport) void _stdcall CopyFrom_Int(VectorInt* destination, const int source[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source, count, destination->begin());
	}
	
	// ベクトルの廃棄
	__declspec(dllexport) void _stdcall Delete_Int(VectorInt* vec, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		delete vec;
	}
}