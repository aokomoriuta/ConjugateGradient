#include"Vector_Double.hpp"
#include"Vector_Int.hpp"
#include<cublas_v2.h>
#include<cusparse_v2.h>

extern "C"
{
	// 方程式を解く
	__declspec(dllexport) void _stdcall Solve(cublasHandle_t* cublas, cusparseHandle_t* cusparse, cusparseMatDescr_t* matDescr, const int deviceID, 
		Vector* elementsVector, VectorInt* rowOffsetsVector, VectorInt* columnIndecesVector,
		Vector* xVector, Vector* bVector,
		Vector* ApVector, Vector* pVector, Vector* rVector,
		const int elementsCount, const int count,
		const double allowableResidual, const int minIteration, const int maxIteration,
		int& iteration, double& residual)
	{
		::cudaSetDevice(deviceID);
		
		const double plusOne = 1;
		const double zero = 0;
		const double minusOne = -1;

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
			cusparseDcsrmv_v2(*cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, *matDescr,
				elements, rowOffsets, columnIndeces, x, &zero, Ap);
			cublasDcopy_v2(*cublas, count, b, 1, r, 1);
			cublasDaxpy_v2(*cublas, count, &minusOne, Ap, 1, r, 1);
			cublasDcopy_v2(*cublas, count, r, 1, p, 1);
			double rr; cublasDdot_v2(*cublas, count, r, 1, r, 1, &rr);

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
				cusparseDcsrmv_v2(*cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, *matDescr,
					elements, rowOffsets, columnIndeces, p, &zero, Ap);
				double pAp; cublasDdot_v2(*cublas, count, p, 1, Ap, 1, &pAp);
				double alpha = rr / pAp;
				double mAlpha = -alpha;
				cublasDaxpy_v2(*cublas, count, &alpha, p, 1, x, 1);
				cublasDaxpy_v2(*cublas, count, &mAlpha, Ap, 1, r, 1);
				double rrNew; cublasDdot_v2(*cublas, count, r, 1, r, 1, &rrNew);
				
				// 収束したかどうかを取得
				residual = sqrt(rrNew);
				converged = (minIteration < iteration) && (residual  < allowableResidual);

				std::cout << iteration << ": " << residual << std::endl;

				// 収束していなかったら
				if(!converged)
				{
					// 残りの計算を実行
					/*
					* β= r'r'/rLDLr
					* p = r' + βp
					*/
					double beta = rrNew / rr;
					//*p = *r + beta * *p;
					cublasDscal_v2(*cublas, count, &beta, p, 1);
					cublasDaxpy_v2(*cublas, count, &plusOne, r, 1, p, 1);

					rr = rrNew;
				}
			}
		}
	}
}