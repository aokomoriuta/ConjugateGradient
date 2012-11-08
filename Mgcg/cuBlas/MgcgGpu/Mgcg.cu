#include"Vector_Double.hpp"
#include"Vector_Int.hpp"
#include<cublas_v2.h>
#include<cusparse_v2.h>

extern "C"
{
	// �s��ƃx�N�g���̐�
	__declspec(dllexport) void _stdcall CsrMV(cusparseHandle_t* cusparse, cusparseMatDescr_t* matDescr, const int deviceID, 
		double* y,
		const double* elements, const int* rowOffsets, const int* columnIndeces,
		const double* x,
		const int elementsCount, const int count,
		const double alpha, const double beta)
	{
		::cudaSetDevice(deviceID);
		::cusparseDcsrmv_v2(*cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &alpha, *matDescr,
			elements, rowOffsets, columnIndeces, x, &beta, y);
	}

	// �x�N�g�����m�̘a
	__declspec(dllexport) void _stdcall Axpy(cublasHandle_t* cublas, const int deviceID, 
		double* y, const double* x, 
		const int count, const double alpha)
	{
		::cudaSetDevice(deviceID);
		::cublasDaxpy_v2(*cublas, count, &alpha, x, 1, y, 1);
	}

	// �x�N�g���̓���
	__declspec(dllexport) double _stdcall Dot(cublasHandle_t* cublas, const int deviceID, 
		double* y, const double* x, 
		const int count)
	{
		::cudaSetDevice(deviceID);

		double result;
		::cublasDdot_v2(*cublas, count, x, 1, y, 1, &result);

		return result;
	}

	// �x�N�g���̃X�J���[�{
	__declspec(dllexport) void _stdcall Scal(cublasHandle_t* cublas, const int deviceID, 
		double* x, double alpha,
		const int count)
	{
		::cudaSetDevice(deviceID);
		::cublasDscal_v2(*cublas, count, &alpha, x, 1);
	}

	// �x�N�g���𕡐�����
	__declspec(dllexport) void _stdcall Copy(cublasHandle_t* cublas, const int deviceID, 
		double* y, const double* x, 
		const int count)
	{
		::cudaSetDevice(deviceID);
		::cublasDcopy_v2(*cublas, count, x, 1, y, 1);
	}

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
			CsrMV(cusparse, matDescr, deviceID, Ap, elements, rowOffsets, columnIndeces, x, elementsCount, count, 1, 0);
			Copy(cublas, deviceID, r, b, count);
			Axpy(cublas, deviceID, r, Ap, count, -1);
			Copy(cublas, deviceID, p, r, count);
			double rr = Dot(cublas, deviceID, r, r, count);

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
				CsrMV(cusparse, matDescr, deviceID, Ap, elements, rowOffsets, columnIndeces, p, elementsCount, count, 1, 0);
				double alpha = rr / Dot(cublas, deviceID, p, Ap, count);
				Axpy(cublas, deviceID, x, p, count, alpha);
				Axpy(cublas, deviceID, r, Ap, count, -alpha);
				double rrNew = Dot(cublas, deviceID, r, r, count);
				
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
					Scal(cublas, deviceID, p, beta, count); Axpy(cublas, deviceID, p, r, count, 1);

					rr = rrNew;
				}
			}
		}
	}
}