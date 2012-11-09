#include"Vector_Double.hpp"
#include"Vector_Int.hpp"
#include<cublas_v2.h>
#include<cusparse_v2.h>
#include<thrust/iterator/constant_iterator.h>

extern "C"
{
	// 行列とベクトルの積
	__declspec(dllexport) void _stdcall CsrMV(cusparseHandle_t* cusparse, cusparseMatDescr_t* matDescr,
		double* y,
		const double* elements, const int* rowOffsets, const int* columnIndeces,
		const double* x,
		const int elementsCount, const int rowCount, const int columnCount,
		const double alpha, const double beta)
	{
		::cusparseDcsrmv_v2(*cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, rowCount, columnCount, elementsCount, &alpha, *matDescr,
			elements, rowOffsets, columnIndeces, x, &beta, y);
	}

	// ベクトル同士の和
	__declspec(dllexport) void _stdcall Axpy(cublasHandle_t* cublas,
		double* y, const double* x,
		const int count, const double alpha)
	{
		::cublasDaxpy_v2(*cublas, count, &alpha, x, 1, y, 1);
	}

	// ベクトルの内積
	__declspec(dllexport) double _stdcall Dot(cublasHandle_t* cublas,
		double* y, const double* x, 
		const int count)
	{
		double result;
		::cublasDdot_v2(*cublas, count, x, 1, y, 1, &result);

		return result;
	}

	// ベクトルのスカラー倍
	__declspec(dllexport) void _stdcall Scal(cublasHandle_t* cublas,
		double* x, double alpha,
		const int count)
	{
		::cublasDscal_v2(*cublas, count, &alpha, x, 1);
	}

	// ベクトルを複製する
	__declspec(dllexport) void _stdcall Copy(cublasHandle_t* cublas,
		double* y, const double* x,
		const int count, const int yOffset, const int xOffset)
	{
		::cublasDcopy_v2(*cublas, count, x + xOffset, 1, y + yOffset, 1);
	}

	// 初期化する
	__declspec(dllexport) void _stdcall Initialize(
		const double elements[], const int rowOffsets[], const int columnIndeces[],
		const double x[], const double b[],
		Vector* elementsVector, VectorInt* rowOffsetsVector, VectorInt* columnIndecesVector,
		Vector* xVector, Vector* bVector,
		Vector* pVector,
		int& minJ, int& maxJ,
		const int count,
		const int countForDevice, const int offsetForDevice, const int elementCountForDevice, const int elementOffsetForDevice)
	{
		// 行列を転送
		thrust::copy_n(elements      + elementOffsetForDevice, elementCountForDevice, elementsVector     ->begin());
		thrust::copy_n(columnIndeces + elementOffsetForDevice, elementCountForDevice, columnIndecesVector->begin());
		thrust::copy_n(rowOffsets    + offsetForDevice,        countForDevice + 1,    rowOffsetsVector   ->begin());

		// 各行の先頭位置を修正
		thrust::transform(rowOffsetsVector->begin(), rowOffsetsVector->begin() + countForDevice + 1, thrust::make_constant_iterator(elementOffsetForDevice), rowOffsetsVector->begin(), thrust::minus<double>());
		
		// ベクトルを転送
		thrust::copy_n(x + offsetForDevice, countForDevice, xVector->begin());
		thrust::copy_n(b + offsetForDevice, countForDevice, bVector->begin());

		// 探索方向ベクトルにデータを転送
		thrust::copy_n(xVector->begin(), countForDevice, pVector->begin() + offsetForDevice);

		// このデバイスが計算する行列の列の範囲を探索
		minJ = *thrust::min_element(columnIndecesVector->begin(), columnIndecesVector->begin() + elementCountForDevice);
		maxJ = *thrust::max_element(columnIndecesVector->begin(), columnIndecesVector->begin() + elementCountForDevice);
	}

	// 探索方向ベクトルの必要部分をホストに転送する
	__declspec(dllexport) void _stdcall P2Host(Vector* pVector, double p[],
		const int thisCount, const int thisOffset,
		const int lastCount, const int nextCount)
	{
		/*    {    last      |     this      |     next     }
		 *    {0, 0, 0, 0, 0 | 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0}
		 * -> {0, 0, 0, 0, 0 | *, *, 0, 0, 0 | 0, 0, 0, 0, 0}
		 * -> {0, 0, 0, 0, 0 | *, *, 0, *, * | 0, 0, 0, 0, 0}
		*/
		thrust::copy_n(pVector->begin() + thisOffset                        , lastCount, p + thisOffset);
		thrust::copy_n(pVector->begin() + thisOffset + thisCount - nextCount, nextCount, p + thisOffset + thisCount - nextCount);
	}

	// 探索方向ベクトルの必要部分をホストから転送する
	__declspec(dllexport) void _stdcall P2Device(Vector* pVector, double p[],
		const int thisCount, const int thisOffset,
		const int lastCount, const int nextCount)
	{
		/*    {    last      |     this      |     next     }
		 *    {0, 0, 0, 0, 0 | *, *, *, *, * | 0, 0, 0, 0, 0}
		 * -> {0, 0, 0, *, * | *, *, *, *, * | 0, 0, 0, 0, 0}
		 * -> {0, 0, 0, *, * | *, *, *, *, * | *, *, 0, 0, 0}
		*/
		thrust::copy_n(p + thisOffset - lastCount, lastCount, pVector->begin() + thisOffset - lastCount);
		thrust::copy_n(p + thisOffset + thisCount, nextCount, pVector->begin() + thisOffset + thisCount);
	}

	// 方程式を解く（初期残差の計算まで）
	__declspec(dllexport) double _stdcall Solve0(cublasHandle_t* cublas, cusparseHandle_t* cusparse, cusparseMatDescr_t* matDescr,
		Vector* elementsVector, VectorInt* rowOffsetsVector, VectorInt* columnIndecesVector,
		Vector* xVector, Vector* bVector,
		Vector* ApVector, Vector* pVector, Vector* rVector,
		const int count,
		const int countForDevice, const int offsetForDevice, const int elementsCountForDevice)
	{
		double* elements = thrust::raw_pointer_cast(&(*elementsVector)[0]);
		int* columnIndeces = thrust::raw_pointer_cast(&(*columnIndecesVector)[0]);
		int* rowOffsets = thrust::raw_pointer_cast(&(*rowOffsetsVector)[0]);
		double* x = thrust::raw_pointer_cast(&(*xVector)[0]);
		double* b = thrust::raw_pointer_cast(&(*bVector)[0]);
		double* Ap = thrust::raw_pointer_cast(&(*ApVector)[0]);
		double* p = thrust::raw_pointer_cast(&(*pVector)[0]);
		double* r = thrust::raw_pointer_cast(&(*rVector)[0]);

		/*
		* (Ap)_0 = A * x
		* r_0 = b - Ap
		* p_0 = r_0
		* rr_0 = r_0・r_0
		*/
		CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, p, elementsCountForDevice, countForDevice, count, 1, 0);
		Copy(cublas, r, b, countForDevice, 0, 0); Axpy(cublas, r, Ap, countForDevice, -1);
		Copy(cublas, p + offsetForDevice, r, countForDevice, 0, 0);
		return Dot(cublas, r, r, countForDevice);
	}

	// 方程式を解く（探索方向係数αの計算まで）
	__declspec(dllexport) double _stdcall Solve1(cublasHandle_t* cublas, cusparseHandle_t* cusparse, cusparseMatDescr_t* matDescr,
		Vector* elementsVector, VectorInt* rowOffsetsVector, VectorInt* columnIndecesVector,
		Vector* ApVector, Vector* pVector,
		const int count,
		const int countForDevice, const int offsetForDevice, const int elementsCountForDevice)
	{
		double* elements = thrust::raw_pointer_cast(&(*elementsVector)[0]);
		int* columnIndeces = thrust::raw_pointer_cast(&(*columnIndecesVector)[0]);
		int* rowOffsets = thrust::raw_pointer_cast(&(*rowOffsetsVector)[0]);
		double* Ap = thrust::raw_pointer_cast(&(*ApVector)[0]);
		double* p = thrust::raw_pointer_cast(&(*pVector)[0]);

		/*
		* Ap = A * p
		* pAp = p・Ap
		*/
		CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, p, elementsCountForDevice, countForDevice, count, 1, 0);
		return Dot(cublas, p + offsetForDevice, Ap, countForDevice);
	}

	// 方程式を解く（現在の残差の計算まで）
	__declspec(dllexport) double _stdcall Solve2(cublasHandle_t* cublas, const double alpha,
		Vector* xVector, 
		Vector* ApVector, Vector* pVector, Vector* rVector,
		const int countForDevice, const int offsetForDevice)
	{
		double* x = thrust::raw_pointer_cast(&(*xVector)[0]);
		double* Ap = thrust::raw_pointer_cast(&(*ApVector)[0]);
		double* p = thrust::raw_pointer_cast(&(*pVector)[0]);
		double* r = thrust::raw_pointer_cast(&(*rVector)[0]);

		/*
		* x' += αp
		* r' -= αAp
		* r'r' = r'・r'
		*/
		Axpy(cublas, x, p + offsetForDevice, countForDevice,  alpha);
		Axpy(cublas, r, Ap                 , countForDevice, -alpha);
		return Dot(cublas, r, r, countForDevice);
	}

	// 方程式を解く（残りの計算すべて）
	__declspec(dllexport) void _stdcall Solve3(cublasHandle_t* cublas, const double beta,
		Vector* pVector, Vector* rVector,
		const int countForDevice, const int offsetForDevice)
	{
		double* p = thrust::raw_pointer_cast(&(*pVector)[0]);
		double* r = thrust::raw_pointer_cast(&(*rVector)[0]);

		/*
		* p = r' + βp
		*/
		Scal(cublas, p + offsetForDevice, beta, countForDevice); Axpy(cublas, p + offsetForDevice, r, countForDevice, 1);
	}

	// 方程式をすべて解く
	__declspec(dllexport) void _stdcall Solve(cublasHandle_t* cublas, cusparseHandle_t* cusparse, cusparseMatDescr_t* matDescr,
		Vector* elementsVector, VectorInt* rowOffsetsVector, VectorInt* columnIndecesVector,
		Vector* xVector, Vector* bVector,
		Vector* ApVector, Vector* pVector, Vector* rVector,
		const int elementsCount, const int count,
		const double allowableResidual, const int minIteration, const int maxIteration,
		int& iteration, double& residual)
	{
		double* elements = thrust::raw_pointer_cast(&(*elementsVector)[0]);
		int* columnIndeces = thrust::raw_pointer_cast(&(*columnIndecesVector)[0]);
		int* rowOffsets = thrust::raw_pointer_cast(&(*rowOffsetsVector)[0]);
		double* x = thrust::raw_pointer_cast(&(*xVector)[0]);
		double* b = thrust::raw_pointer_cast(&(*bVector)[0]);
		double* Ap = thrust::raw_pointer_cast(&(*ApVector)[0]);
		double* p = thrust::raw_pointer_cast(&(*pVector)[0]);
		double* r = thrust::raw_pointer_cast(&(*rVector)[0]);
		{
			// 初期値を設定
			/*
			* (Ap)_0 = A * x
			* r_0 = b - Ap
			* rr_0 = r_0・r_0
			* p_0 = r_0
			*/
			CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, x, elementsCount, count, count, 1, 0);
			Copy(cublas, r, b, count, 0, 0); Axpy(cublas, r, Ap, count, -1);
			double rr = Dot(cublas, r, r, count);
			Copy(cublas, p, r, count, 0, 0);

			// 収束したかどうか
			bool converged = false;

			// 収束しない間繰り返す
			for(iteration = 0; !converged; iteration++)
			{
				// 計算を実行
				/*
				* Ap = A * p
				* α = rr/(p・Ap)
				* x' += αp
				* r' -= αAp
				* r'r' = r'・r'
				*/
				CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, p, elementsCount, count, count, 1, 0);
				double alpha = rr / Dot(cublas, p, Ap, count);
				Axpy(cublas, x, p, count, alpha);
				Axpy(cublas, r, Ap, count, -alpha);
				double rrNew = Dot(cublas, r, r, count);
				
				// 収束したかどうかを取得
				residual = sqrt(rrNew);
				converged = (minIteration <= iteration) && (residual  < allowableResidual);

				//std::cout << iteration << ": " << residual << std::endl;

				// 収束していなかったら
				if(!converged)
				{
					// 残りの計算を実行
					/*
					* β= r'r'/rLDLr
					* p = r' + βp
					*/
					double beta = rrNew / rr;
					Scal(cublas, p, beta, count); Axpy(cublas, p, r, count, 1);
					rr = rrNew;
				}
			}
		}
	}
}