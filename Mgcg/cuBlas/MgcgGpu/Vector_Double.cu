#include"Vector_Double.hpp"
#include<thrust/copy.h>

extern "C"
{
	// ベクトルの作成
	__declspec(dllexport) Vector* _stdcall Create_Double(const int size, const int deviceID)
	{
		::cudaSetDevice(deviceID);

		Vector* vec = new Vector(size);

		return vec;
	}

	// 配列へ複製
	__declspec(dllexport) void _stdcall CopyTo_Double(const Vector* source, double destination[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source->begin(), count, destination);
	}

	// 配列から複製
	__declspec(dllexport) void _stdcall CopyFrom_Double(Vector* destination, const double source[], const int count, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		thrust::copy_n(source, count, destination->begin());
	}
	
	// ベクトルの廃棄
	__declspec(dllexport) void _stdcall Delete_Double(Vector* vec, const int deviceID)
	{
		::cudaSetDevice(deviceID);
		delete vec;
	}
}