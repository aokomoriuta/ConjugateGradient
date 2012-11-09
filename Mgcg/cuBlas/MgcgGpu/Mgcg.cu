#include"Vector_Double.hpp"
#include"Vector_Int.hpp"
#include<cublas_v2.h>
#include<cusparse_v2.h>
#include<thrust/iterator/constant_iterator.h>

extern "C"
{
	// �s��ƃx�N�g���̐�
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

	// �x�N�g�����m�̘a
	__declspec(dllexport) void _stdcall Axpy(cublasHandle_t* cublas,
		double* y, const double* x,
		const int count, const double alpha)
	{
		::cublasDaxpy_v2(*cublas, count, &alpha, x, 1, y, 1);
	}

	// �x�N�g���̓���
	__declspec(dllexport) double _stdcall Dot(cublasHandle_t* cublas,
		double* y, const double* x, 
		const int count)
	{
		double result;
		::cublasDdot_v2(*cublas, count, x, 1, y, 1, &result);

		return result;
	}

	// �x�N�g���̃X�J���[�{
	__declspec(dllexport) void _stdcall Scal(cublasHandle_t* cublas,
		double* x, double alpha,
		const int count)
	{
		::cublasDscal_v2(*cublas, count, &alpha, x, 1);
	}

	// �x�N�g���𕡐�����
	__declspec(dllexport) void _stdcall Copy(cublasHandle_t* cublas,
		double* y, const double* x,
		const int count, const int yOffset, const int xOffset)
	{
		::cublasDcopy_v2(*cublas, count, x + xOffset, 1, y + yOffset, 1);
	}

	// ����������
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
		// �s���]��
		thrust::copy_n(elements      + elementOffsetForDevice, elementCountForDevice, elementsVector     ->begin());
		thrust::copy_n(columnIndeces + elementOffsetForDevice, elementCountForDevice, columnIndecesVector->begin());
		thrust::copy_n(rowOffsets    + offsetForDevice,        countForDevice + 1,    rowOffsetsVector   ->begin());

		// �e�s�̐擪�ʒu���C��
		thrust::transform(rowOffsetsVector->begin(), rowOffsetsVector->begin() + countForDevice + 1, thrust::make_constant_iterator(elementOffsetForDevice), rowOffsetsVector->begin(), thrust::minus<double>());
		
		// �x�N�g����]��
		thrust::copy_n(x + offsetForDevice, countForDevice, xVector->begin());
		thrust::copy_n(b + offsetForDevice, countForDevice, bVector->begin());

		// �T�������x�N�g���Ƀf�[�^��]��
		thrust::copy_n(xVector->begin(), countForDevice, pVector->begin() + offsetForDevice);

		// ���̃f�o�C�X���v�Z����s��̗�͈̔͂�T��
		minJ = *thrust::min_element(columnIndecesVector->begin(), columnIndecesVector->begin() + elementCountForDevice);
		maxJ = *thrust::max_element(columnIndecesVector->begin(), columnIndecesVector->begin() + elementCountForDevice);
	}

	// �T�������x�N�g���̕K�v�������z�X�g�ɓ]������
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

	// �T�������x�N�g���̕K�v�������z�X�g����]������
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

	// �������������i�����c���̌v�Z�܂Łj
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
		* rr_0 = r_0�Er_0
		*/
		CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, p, elementsCountForDevice, countForDevice, count, 1, 0);
		Copy(cublas, r, b, countForDevice, 0, 0); Axpy(cublas, r, Ap, countForDevice, -1);
		Copy(cublas, p + offsetForDevice, r, countForDevice, 0, 0);
		return Dot(cublas, r, r, countForDevice);
	}

	// �������������i�T�������W�����̌v�Z�܂Łj
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
		* pAp = p�EAp
		*/
		CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, p, elementsCountForDevice, countForDevice, count, 1, 0);
		return Dot(cublas, p + offsetForDevice, Ap, countForDevice);
	}

	// �������������i���݂̎c���̌v�Z�܂Łj
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
		* x' += ��p
		* r' -= ��Ap
		* r'r' = r'�Er'
		*/
		Axpy(cublas, x, p + offsetForDevice, countForDevice,  alpha);
		Axpy(cublas, r, Ap                 , countForDevice, -alpha);
		return Dot(cublas, r, r, countForDevice);
	}

	// �������������i�c��̌v�Z���ׂāj
	__declspec(dllexport) void _stdcall Solve3(cublasHandle_t* cublas, const double beta,
		Vector* pVector, Vector* rVector,
		const int countForDevice, const int offsetForDevice)
	{
		double* p = thrust::raw_pointer_cast(&(*pVector)[0]);
		double* r = thrust::raw_pointer_cast(&(*rVector)[0]);

		/*
		* p = r' + ��p
		*/
		Scal(cublas, p + offsetForDevice, beta, countForDevice); Axpy(cublas, p + offsetForDevice, r, countForDevice, 1);
	}

	// �����������ׂĉ���
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
			// �����l��ݒ�
			/*
			* (Ap)_0 = A * x
			* r_0 = b - Ap
			* rr_0 = r_0�Er_0
			* p_0 = r_0
			*/
			CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, x, elementsCount, count, count, 1, 0);
			Copy(cublas, r, b, count, 0, 0); Axpy(cublas, r, Ap, count, -1);
			double rr = Dot(cublas, r, r, count);
			Copy(cublas, p, r, count, 0, 0);

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
				CsrMV(cusparse, matDescr, Ap, elements, rowOffsets, columnIndeces, p, elementsCount, count, count, 1, 0);
				double alpha = rr / Dot(cublas, p, Ap, count);
				Axpy(cublas, x, p, count, alpha);
				Axpy(cublas, r, Ap, count, -alpha);
				double rrNew = Dot(cublas, r, r, count);
				
				// �����������ǂ������擾
				residual = sqrt(rrNew);
				converged = (minIteration <= iteration) && (residual  < allowableResidual);

				//std::cout << iteration << ": " << residual << std::endl;

				// �������Ă��Ȃ�������
				if(!converged)
				{
					// �c��̌v�Z�����s
					/*
					* ��= r'r'/rLDLr
					* p = r' + ��p
					*/
					double beta = rrNew / rr;
					Scal(cublas, p, beta, count); Axpy(cublas, p, r, count, 1);
					rr = rrNew;
				}
			}
		}
	}
}