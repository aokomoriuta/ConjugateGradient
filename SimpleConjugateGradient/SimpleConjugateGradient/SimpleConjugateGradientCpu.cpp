#include<boost/timer.hpp>
#include<boost/numeric/ublas/vector.hpp>
#include<boost/numeric/ublas/matrix_sparse.hpp>

#include<iostream>
#include<vector>

void SolveCpu(
	boost::numeric::ublas::compressed_matrix<double> A, boost::numeric::ublas::vector<double>& x, boost::numeric::ublas::vector<double> b,
	const int elementsCount, const int count,
	const double allowableResidual, const int minIteration, const int maxIteration,
	int& iteration, double& residual)
{
	using namespace boost::numeric::ublas;

	boost::numeric::ublas::vector<double> Ap(count);
	boost::numeric::ublas::vector<double> r(count);
	boost::numeric::ublas::vector<double> p(count);

	// �����l��ݒ�
	/*
	* (Ap)_0 = A * x
	* r_0 = b - Ap
	* rr_0 = r_0�Er_0
	* p_0 = r_0
	*/
	Ap = prod(A, x);
	r = b - Ap;
	double rr = inner_prod(r, r);
	p = r;

	// �����������ǂ���
	bool converged = false;
	
		Ap = prod(A, p);
}

int main()
{
	const int N = 10;
	const double ALLOWABLE_RESIDUAL = 1e-8;

	const int MIN_ITERATION = 0;
	const int MAX_ITERATION = N;

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
	// CPU���̔z����m��
	boost::numeric::ublas::compressed_matrix<double> A(N, N, nonZeroCount);
	boost::numeric::ublas::vector<double> bDevice(b.size());
	boost::numeric::ublas::vector<double> xDevice(x.size());

	// CPU���z��֓��͒l�i�s��ƃx�N�g���j�𕡐�
	A.reserve(nonZeroCount, false);
	std::copy(elements.begin(),      elements.end(),      A.value_data().begin());
	std::copy(rowOffsets.begin(),    rowOffsets.end(),    A.index1_data().begin());
	std::copy(columnIndeces.begin(), columnIndeces.end(), A.index2_data().begin());
	A.set_filled(N+1, nonZeroCount);
	std::copy(b.begin(), b.end(), bDevice.begin());

	
	/********************************/
	/********** �v�Z�����s **********/
	/********************************/
	boost::timer timer;

	// GPU�ŉ���
	int iterationGpu = 0;
	double residualGpu = 0;
	
	timer.restart();

	SolveCpu(A, xDevice, bDevice,
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
	std::copy(xDevice.begin(), xDevice.end(), x.begin());

	// ���ʂ̕\��
	for(int i = 0; i < N; i++)
	{
		std::cout << x[i] << std::endl;
	}

	return 0;
}