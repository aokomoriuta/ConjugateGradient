#include"Vector_Int.hpp"
#include<thrust/copy.h>
#include<thrust/transform.h>


extern "C"
{
	// ベクトルの作成
	__declspec(dllexport) VectorInt* _stdcall Create_Int(const int size)
	{
		VectorInt* vec = new VectorInt(size);

		return vec;
	}

	// 配列へ複製
	__declspec(dllexport) void _stdcall CopyToArray_Int(const VectorInt* source, int destination[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source->begin() + sourceOffset, count, destination + destinationOffset);
	}

	// 配列から複製
	__declspec(dllexport) void _stdcall CopyFromArray_Int(VectorInt* destination, int source[],
		const int count, const int sourceOffset, const int destinationOffset)
	{
		thrust::copy_n(source + sourceOffset, count, destination->begin() + destinationOffset);
	}
	
	// ベクトルの廃棄
	__declspec(dllexport) void _stdcall Delete_Int(VectorInt* vec)
	{
		delete vec;
	}
	
	// CUDA用ポインタへ変換する
	__declspec(dllexport) int* _stdcall ToRawPtr_Int(VectorInt* vec)
	{
		return thrust::raw_pointer_cast(&(*vec)[0]);
	}
}