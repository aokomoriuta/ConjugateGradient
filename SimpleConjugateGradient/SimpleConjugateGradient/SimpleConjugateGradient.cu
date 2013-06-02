#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<cusparse_v2.h>

#include<thrust/device_vector.h>
#include<thrust/copy.h>

#include<boost/timer.hpp>

#include<iostream>

void SolveGpu(
	thrust::device_vector<double> elementsVector, thrust::device_vector<int> rowOffsetsVector, thrust::device_vector<int> columnIndecesVector,
	thrust::device_vector<double>& xVector, thrust::device_vector<double> bVector,
	const int elementsCount, const int count,
	const double allowableResidual, const int minIteration, const int maxIteration,
	int& iteration, double& residual)
{
	/********************************************/
	/********** CUBLAS��cuSPARSE�̏��� **********/
	/********************************************/
	// CUBLAS�n���h�����쐬
	::cublasHandle_t cublas;
	::cublasCreate(&cublas);

	// cuSPARSE�n���h�����쐬
	::cusparseHandle_t cusparse;
	::cusparseCreate(&cusparse);

	// �s��`�����쐬
	// * ��ʓI�Ȍ`��
	// * �ԍ���0����J�n
	::cusparseMatDescr_t matDescr;
	::cusparseCreateMatDescr(&matDescr);
	::cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	::cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

	// cusparse�Ŏg���萔�Q
	const double plusOne = 1;
	const double zero = 0;
	const double minusOne = -1;

	
	/********************************/
	/********** �z��̏��� **********/
	/********************************/
	// Ap, p, r���쐬
	thrust::device_vector<double> ApVector(count);
	thrust::device_vector<double> pVector(count);
	thrust::device_vector<double> rVector(count);

	// ���m���x�N�g����0�ŏ�����
	thrust::fill_n(xVector.begin(), count, 0.0);

	// CUDA�|�C���^�ɕϊ�
	double* elements = thrust::raw_pointer_cast(&elementsVector[0]);
	int* columnIndeces = thrust::raw_pointer_cast(&columnIndecesVector[0]);
	int* rowOffsets = thrust::raw_pointer_cast(&rowOffsetsVector[0]);
	double* x = thrust::raw_pointer_cast(&xVector[0]);
	double* b = thrust::raw_pointer_cast(&bVector[0]);
	double* Ap = thrust::raw_pointer_cast(&ApVector[0]);
	double* p = thrust::raw_pointer_cast(&pVector[0]);
	double* r = thrust::raw_pointer_cast(&rVector[0]);

	/****************************************/
	/********** �������z�@ **********/
	/****************************************/
	// �����l��ݒ�
	/*
	* (Ap)_0 = A * x
	* r_0 = b - Ap
	* rr_0 = r_0�Er_0
	* p_0 = r_0
	*/
	cusparseDcsrmv_v2(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, matDescr,
		elements, rowOffsets, columnIndeces, x, &zero, Ap);
	cublasDcopy_v2(cublas, count, b, 1, r, 1);
	cublasDaxpy_v2(cublas, count, &minusOne, Ap, 1, r, 1);
	cublasDcopy_v2(cublas, count, r, 1, p, 1);
	double rr; cublasDdot_v2(cublas, count, r, 1, r, 1, &rr);

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
		cusparseDcsrmv_v2(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, matDescr,
			elements, rowOffsets, columnIndeces, p, &zero, Ap);
		double pAp; cublasDdot_v2(cublas, count, p, 1, Ap, 1, &pAp);
		double alpha = rr / pAp;
		double mAlpha = -alpha;
		cublasDaxpy_v2(cublas, count, &alpha, p, 1, x, 1);
		cublasDaxpy_v2(cublas, count, &mAlpha, Ap, 1, r, 1);
		double rrNew; cublasDdot_v2(cublas, count, r, 1, r, 1, &rrNew);
				
		// �����������ǂ������擾
		residual = sqrt(rrNew);
		converged = (minIteration < iteration) && (residual  < allowableResidual);

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
			cublasDscal_v2(cublas, count, &beta, p, 1);
			cublasDaxpy_v2(cublas, count, &plusOne, r, 1, p, 1);

			rr = rrNew;
		}
	}
}

int main()
{
	const int N = 1024 * 64;
	const double ALLOWABLE_RESIDUAL = 1e-8;

	const int MIN_ITERATION = 0;
	const int MAX_ITERATION = N;

	std::cout << "Solving liner equations with " << N  << " unknown variables" << std::endl;

	/**********************************/
	/********** ���͒l�̏��� **********/
	/**********************************/

	// CSR�`���a�s��̃f�[�^
	//* �v�f�̒l
	//* ��ԍ�
	//* �e�s�̐擪�ʒu
	std::vector<double> elements(N*3);
	std::vector<int> columnIndeces(N*3);
	std::vector<int> rowOffsets(N+1);

	// ���������s�����������
	//�i�Ίp����2�ł��ׂ̗�1�ɂȂ�A������Ȃ�j
	// | 2 1 0 0 0 0 0 0 �E�E�E 0 0 0|
	// | 1 2 1 0 0 0 0 0 �E�E�E 0 0 0|
	// | 0 1 2 1 0 0 0 0 �E�E�E 0 0 0|
	// | 0 0 1 2 1 0 0 0 �E�E�E 0 0 0|
	// | 0 0 0 1 2 1 0 0 �E�E�E 0 0 0|
	// | 0 0 0 0 1 2 1 0 �E�E�E 0 0 0|
	// | 0 0 0 0 0 1 2 1 �E�E�E 0 0 0|
	// | 0 0 0 0 0 0 1 2 �E�E�E 0 0 0|
	// | 0 0 0 0 0 0 0 0 �E�E�E 2 1 0|
	// | 0 0 0 0 0 0 0 0 �E�E�E 1 2 1|
	// | 0 0 0 0 0 0 0 0 �E�E�E 0 1 2|
	int nonZeroCount = 0;
	rowOffsets[0] = 0;
	for(int i = 0; i < N; i++)
	{
		// �Ίp��
		elements[nonZeroCount] = 2;
		columnIndeces[nonZeroCount] = i;
		nonZeroCount++;

		// �Ίp���̍���
		if(i > 0)
		{
			elements[nonZeroCount] = 1;
			columnIndeces[nonZeroCount] = i - 1;
			nonZeroCount++;
		}

		// �Ίp���̉E��
		if(i < N-1)
		{
			elements[nonZeroCount] = 1;
			columnIndeces[nonZeroCount] = i + 1;
			nonZeroCount++;
		}

		// ���̍s�̐擪�ʒu
		rowOffsets[i+1] = nonZeroCount;
	}

	// �E�Ӄx�N�g���𐶐�
	std::vector<double> b(N);
	for(int i = 0; i < N; i++)
	{
		b[i] = i * i * 0.5;
	}

	// ���m���x�N�g���𐶐�
	std::vector<double> x(N);

	/**********************************/
	/********** ���͒l�̓]�� **********/
	/**********************************/
	// GPU���̔z����m��
	// �i�|�C���^�Ǘ����ʓ|�Ȃ̂�thrust�g���ƕ֗��I�j
	thrust::device_vector<double> elementsDevice(elements.size());
	thrust::device_vector<int>    columnIndecesDevice(columnIndeces.size());
	thrust::device_vector<int>    rowOffsetsDevice(rowOffsets.size());
	thrust::device_vector<double> bDevice(b.size());
	thrust::device_vector<double> xDevice(x.size());

	// GPU���z��֓��͒l�i�s��ƃx�N�g���j�𕡐�
	thrust::copy_n(elements.begin(),      N*3, elementsDevice.begin());
	thrust::copy_n(columnIndeces.begin(), N*3, columnIndecesDevice.begin());
	thrust::copy_n(rowOffsets.begin(),    N+1, rowOffsetsDevice.begin());
	thrust::copy_n(b.begin(), N, bDevice.begin());

	
	/********************************/
	/********** �v�Z�����s **********/
	/********************************/
	boost::timer timer;

	// GPU�ŉ���
	int iterationGpu = 0;
	double residualGpu = 0;
	
	timer.restart();

	SolveGpu(elementsDevice, rowOffsetsDevice, columnIndecesDevice,
		xDevice, bDevice,
		nonZeroCount, N, ALLOWABLE_RESIDUAL, MIN_ITERATION, MAX_ITERATION,
		iterationGpu, residualGpu);

	std::cout << "GPU:" << std::endl
	          << "	Iteration: " << iterationGpu << std::endl
			  << "	Residual: " << residualGpu << std::endl
			  << "	Time: " << timer.elapsed() << "[s]" << std::endl;

	/************************************/
	/********** �v�Z���ʂ��擾 **********/
	/************************************/
	// GPU���z�񂩂猋�ʂ𕡐�
	thrust::copy_n(xDevice.begin(), N, x.begin());

	// ���ʂ̕\��
	for(int i = 0; i < N; i++)
	{
		//std::cout << x[i] << std::endl;
	}

	return 0;
}