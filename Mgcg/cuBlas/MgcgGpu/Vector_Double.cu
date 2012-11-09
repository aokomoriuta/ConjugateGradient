#include"Vector_Double.hpp"
#include<thrust/copy.h>

extern "C"
{
	// ベクトルの作成
	__declspec(dllexport) Vector* _stdcall Create_Double(const int size)
	{
		Vector* vec = new Vector(size);

		return vec;
	}

	// 配列へ複製
	__declspec(dllexport) void _stdcall CopyToArray_Double(const Vector* source, double destination[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source->begin() + sourceOffset, count, destination + destinationOffset);
	}

	// 配列から複製
	__declspec(dllexport) void _stdcall CopyFromArray_Double(Vector* destination, const double source[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source + sourceOffset, count, destination->begin() + destinationOffset);
	}
	
	// ベクトルの廃棄
	__declspec(dllexport) void _stdcall Delete_Double(Vector* vec)
	{
		delete vec;
	}
	
	// CUDA用ポインタへ変換する
	__declspec(dllexport) double* _stdcall ToRawPtr_Double(Vector* vec)
	{
		return thrust::raw_pointer_cast(&(*vec)[0]);
	}

	// デバイス間でデータを転送する
	__declspec(dllexport) void _stdcall CopyFromDevice_Double(const double* source, double* destination,
		const int count, const int sourceOffset, const int destinationOffset)
	{
		::cudaMemcpy(destination + destinationOffset, source + sourceOffset, count, cudaMemcpyDeviceToDevice);
	}
}