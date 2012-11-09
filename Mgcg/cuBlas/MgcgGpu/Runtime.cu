#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<cusparse_v2.h>

extern "C"
{
	__declspec(dllexport) int _stdcall GetDeviceCount()
	{
		int count;
		::cudaGetDeviceCount(&count);

		return count;
	}

	__declspec(dllexport) void _stdcall SetDevice(const int deviceID)
	{		
		::cudaSetDevice(deviceID);
	}

	__declspec(dllexport) cublasHandle_t* _stdcall CreateBlas()
	{
		::cublasHandle_t* handle = new cublasHandle_t();
		::cublasCreate_v2(handle);

		return handle;
	}

	__declspec(dllexport) void _stdcall DestroyBlas(cublasHandle_t* cublas)
	{		
		::cublasDestroy_v2(*cublas);
		delete cublas;
	}

	__declspec(dllexport) cusparseHandle_t* _stdcall CreateSparse()
	{
		::cusparseHandle_t* handle = new cusparseHandle_t();
		::cusparseCreate(handle);

		return handle;
	}

	__declspec(dllexport) void _stdcall DestroySparse(cusparseHandle_t* cusparse)
	{		
		::cusparseDestroy(*cusparse);
		delete cusparse;
	}

	__declspec(dllexport) cusparseMatDescr_t* _stdcall CreateMatDescr()
	{
		::cusparseMatDescr_t* matDescr = new cusparseMatDescr_t();
		cusparseCreateMatDescr(matDescr);
		cusparseSetMatType(*matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(*matDescr, CUSPARSE_INDEX_BASE_ZERO);

		return matDescr;
	}

	__declspec(dllexport) void _stdcall DestroyMatDescr(cusparseMatDescr_t* matDescr)
	{		
		::cusparseDestroyMatDescr(*matDescr);
		delete matDescr;
	}
}