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

	// 初期値を設定
	/*
	* (Ap)_0 = A * x
	* r_0 = b - Ap
	* rr_0 = r_0・r_0
	* p_0 = r_0
	*/
	Ap = prod(A, x);
	r = b - Ap;
	double rr = inner_prod(r, r);
	p = r;

	// 収束したかどうか
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
	/********** 入力値の準備 **********/
	/**********************************/

	// CSR形式疎行列のデータ
	//* 要素の値
	//* 列番号
	//* 各行の先頭位置
	std::vector<double> elements(N*3);
	std::vector<int> columnIndeces(N*3);
	std::vector<int> rowOffsets(N+1);

	// 中央差分行列を準備する
	//（対角項が2でその隣が1になる、↓こんなやつ）
	// | 2 1 0 0 0 0 0 0 ・・・ 0 0 0|
	// | 1 2 1 0 0 0 0 0 ・・・ 0 0 0|
	// | 0 1 2 1 0 0 0 0 ・・・ 0 0 0|
	// | 0 0 1 2 1 0 0 0 ・・・ 0 0 0|
	// | 0 0 0 1 2 1 0 0 ・・・ 0 0 0|
	// | 0 0 0 0 1 2 1 0 ・・・ 0 0 0|
	// | 0 0 0 0 0 1 2 1 ・・・ 0 0 0|
	// | 0 0 0 0 0 0 1 2 ・・・ 0 0 0|
	// | 0 0 0 0 0 0 0 0 ・・・ 2 1 0|
	// | 0 0 0 0 0 0 0 0 ・・・ 1 2 1|
	// | 0 0 0 0 0 0 0 0 ・・・ 0 1 2|
	int nonZeroCount = 0;
	rowOffsets[0] = 0;
	for(int i = 0; i < N; i++)
	{
		// 対角項
		elements[nonZeroCount] = 2;
		columnIndeces[nonZeroCount] = i;
		nonZeroCount++;

		// 対角項の左隣
		if(i > 0)
		{
			elements[nonZeroCount] = 1;
			columnIndeces[nonZeroCount] = i - 1;
			nonZeroCount++;
		}

		// 対角項の右隣
		if(i < N-1)
		{
			elements[nonZeroCount] = 1;
			columnIndeces[nonZeroCount] = i + 1;
			nonZeroCount++;
		}

		// 次の行の先頭位置
		rowOffsets[i+1] = nonZeroCount;
	}

	// 右辺ベクトルを生成
	std::vector<double> b(N);
	for(int i = 0; i < N; i++)
	{
		b[i] = i * i * 0.5;
	}

	// 未知数ベクトルを生成
	std::vector<double> x(N);

	/**********************************/
	/********** 入力値の転送 **********/
	/**********************************/
	// CPU側の配列を確保
	boost::numeric::ublas::compressed_matrix<double> A(N, N, nonZeroCount);
	boost::numeric::ublas::vector<double> bDevice(b.size());
	boost::numeric::ublas::vector<double> xDevice(x.size());

	// CPU側配列へ入力値（行列とベクトル）を複製
	A.reserve(nonZeroCount, false);
	std::copy(elements.begin(),      elements.end(),      A.value_data().begin());
	std::copy(rowOffsets.begin(),    rowOffsets.end(),    A.index1_data().begin());
	std::copy(columnIndeces.begin(), columnIndeces.end(), A.index2_data().begin());
	A.set_filled(N+1, nonZeroCount);
	std::copy(b.begin(), b.end(), bDevice.begin());

	
	/********************************/
	/********** 計算を実行 **********/
	/********************************/
	boost::timer timer;

	// GPUで解く
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
	/********** 計算結果を取得 **********/
	/************************************/
	// GPU側配列から結果を複製
	std::copy(xDevice.begin(), xDevice.end(), x.begin());

	// 結果の表示
	for(int i = 0; i < N; i++)
	{
		std::cout << x[i] << std::endl;
	}

	return 0;
}