#include"Vector_Double.hpp"
#include"Vector_Int.hpp"
#include<cublas_v2.h>
#include<cusparse_v2.h>

extern "C"
{
	// ������������
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
			// �����l��ݒ�
			/*
			* (Ap)_0 = A * x
			* r_0 = b - Ap
			* rr_0 = r_0�Er_0
			* p_0 = r_0
			*/
			cusparseDcsrmv_v2(*cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, *matDescr,
				elements, rowOffsets, columnIndeces, x, &zero, Ap);
			cublasDcopy_v2(*cublas, count, b, 1, r, 1);
			cublasDaxpy_v2(*cublas, count, &minusOne, Ap, 1, r, 1);
			cublasDcopy_v2(*cublas, count, r, 1, p, 1);
			double rr; cublasDdot_v2(*cublas, count, r, 1, r, 1, &rr);

			// �����������ǂ���
			bool converged = false;

			// �������Ȃ��ԌJ��Ԃ�
			for(iteration = 0; !converged; iteration++)
			{
				// �v�Z�����s
				/*
				* Ap = A * p
				* �� = rr/(p�EAp)
				* x' += ��p
				* r' -= ��Ap
				* r'r' = r'�Er'
				*/
				cusparseDcsrmv_v2(*cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, *matDescr,
					elements, rowOffsets, columnIndeces, p, &zero, Ap);
				double pAp; cublasDdot_v2(*cublas, count, p, 1, Ap, 1, &pAp);
				double alpha = rr / pAp;
				double mAlpha = -alpha;
				cublasDaxpy_v2(*cublas, count, &alpha, p, 1, x, 1);
				cublasDaxpy_v2(*cublas, count, &mAlpha, Ap, 1, r, 1);
				double rrNew; cublasDdot_v2(*cublas, count, r, 1, r, 1, &rrNew);
				
				// �����������ǂ������擾
				residual = sqrt(rrNew);
				converged = (minIteration < iteration) && (residual  < allowableResidual);

				std::cout << iteration << ": " << residual << std::endl;

				// �������Ă��Ȃ�������
				if(!converged)
				{
					// �c��̌v�Z�����s
					/*
					* ��= r'r'/rLDLr
					* p = r' + ��p
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