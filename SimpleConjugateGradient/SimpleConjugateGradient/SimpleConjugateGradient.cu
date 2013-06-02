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
	/********** CUBLASとcuSPARSEの準備 **********/
	/********************************************/
	// CUBLASハンドルを作成
	::cublasHandle_t cublas;
	::cublasCreate(&cublas);

	// cuSPARSEハンドルを作成
	::cusparseHandle_t cusparse;
	::cusparseCreate(&cusparse);

	// 行列形式を作成
	// * 一般的な形式
	// * 番号は0から開始
	::cusparseMatDescr_t matDescr;
	::cusparseCreateMatDescr(&matDescr);
	::cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	::cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

	// cusparseで使う定数群
	const double plusOne = 1;
	const double zero = 0;
	const double minusOne = -1;

	
	/********************************/
	/********** 配列の準備 **********/
	/********************************/
	// Ap, p, rを作成
	thrust::device_vector<double> ApVector(count);
	thrust::device_vector<double> pVector(count);
	thrust::device_vector<double> rVector(count);

	// 未知数ベクトルを0で初期化
	thrust::fill_n(xVector.begin(), count, 0.0);

	// CUDAポインタに変換
	double* elements = thrust::raw_pointer_cast(&elementsVector[0]);
	int* columnIndeces = thrust::raw_pointer_cast(&columnIndecesVector[0]);
	int* rowOffsets = thrust::raw_pointer_cast(&rowOffsetsVector[0]);
	double* x = thrust::raw_pointer_cast(&xVector[0]);
	double* b = thrust::raw_pointer_cast(&bVector[0]);
	double* Ap = thrust::raw_pointer_cast(&ApVector[0]);
	double* p = thrust::raw_pointer_cast(&pVector[0]);
	double* r = thrust::raw_pointer_cast(&rVector[0]);

	/****************************************/
	/********** 共役勾配法 **********/
	/****************************************/
	// 初期値を設定
	/*
	* (Ap)_0 = A * x
	* r_0 = b - Ap
	* rr_0 = r_0・r_0
	* p_0 = r_0
	*/
	cusparseDcsrmv_v2(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, matDescr,
		elements, rowOffsets, columnIndeces, x, &zero, Ap);
	cublasDcopy_v2(cublas, count, b, 1, r, 1);
	cublasDaxpy_v2(cublas, count, &minusOne, Ap, 1, r, 1);
	cublasDcopy_v2(cublas, count, r, 1, p, 1);
	double rr; cublasDdot_v2(cublas, count, r, 1, r, 1, &rr);

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
		cusparseDcsrmv_v2(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, count, count, elementsCount, &plusOne, matDescr,
			elements, rowOffsets, columnIndeces, p, &zero, Ap);
		double pAp; cublasDdot_v2(cublas, count, p, 1, Ap, 1, &pAp);
		double alpha = rr / pAp;
		double mAlpha = -alpha;
		cublasDaxpy_v2(cublas, count, &alpha, p, 1, x, 1);
		cublasDaxpy_v2(cublas, count, &mAlpha, Ap, 1, r, 1);
		double rrNew; cublasDdot_v2(cublas, count, r, 1, r, 1, &rrNew);
				
		// 収束したかどうかを取得
		residual = sqrt(rrNew);
		converged = (minIteration < iteration) && (residual  < allowableResidual);

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
	// GPU側の配列を確保
	// （ポインタ管理が面倒なのでthrust使うと便利！）
	thrust::device_vector<double> elementsDevice(elements.size());
	thrust::device_vector<int>    columnIndecesDevice(columnIndeces.size());
	thrust::device_vector<int>    rowOffsetsDevice(rowOffsets.size());
	thrust::device_vector<double> bDevice(b.size());
	thrust::device_vector<double> xDevice(x.size());

	// GPU側配列へ入力値（行列とベクトル）を複製
	thrust::copy_n(elements.begin(),      N*3, elementsDevice.begin());
	thrust::copy_n(columnIndeces.begin(), N*3, columnIndecesDevice.begin());
	thrust::copy_n(rowOffsets.begin(),    N+1, rowOffsetsDevice.begin());
	thrust::copy_n(b.begin(), N, bDevice.begin());

	
	/********************************/
	/********** 計算を実行 **********/
	/********************************/
	boost::timer timer;

	// GPUで解く
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
	/********** 計算結果を取得 **********/
	/************************************/
	// GPU側配列から結果を複製
	thrust::copy_n(xDevice.begin(), N, x.begin());

	// 結果の表示
	for(int i = 0; i < N; i++)
	{
		//std::cout << x[i] << std::endl;
	}

	return 0;
}