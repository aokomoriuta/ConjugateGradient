#include "ComputerCpu.hpp"

typedef boost::numeric::ublas::vector<double> Vector;
typedef boost::numeric::ublas::compressed_matrix<double> SparseMatrix;

namespace LWisteria{ namespace Mgcg
{
#pragma unmanaged
	ComputerCpuNative::ComputerCpuNative(int n)
	{
		// 要素数を設定
		count = n;

		// ベクトルと行列を生成
		result = std::shared_ptr<Vector>(new Vector(count));
		matrix = std::shared_ptr<SparseMatrix>(new SparseMatrix(count, count));
		vector = std::shared_ptr<Vector>(new Vector(count));
	}

	void ComputerCpuNative::Write(double matrixPtr[], double vectorPtr[])
	{
		(*matrix)(0, 0) = 5;
	}

	void ComputerCpuNative::Matrix_x_Vector()
	{
	}

	void ComputerCpuNative::Read(double resultPtr[])
	{
	}
#pragma managed
	ComputerCpu::ComputerCpu(int n)
	{
		// 計算クラスを作成
		computer = new ComputerCpuNative(n);
	}
	
	ComputerCpu::~ComputerCpu()
	{
		// 計算クラスを廃棄
		delete(computer);
	}
	
	void ComputerCpu::Write(array<double>^ left, array<double>^ right)
	{
		// 先頭ポインタ取得
		pin_ptr<double> leftPtr = &left[0];
		pin_ptr<double> rightPtr = &right[0];

		// 書き込み
		computer->Write(leftPtr, rightPtr);
	}

	void ComputerCpu::Matrix_x_Vector()
	{
		// 演算実行
		computer->Matrix_x_Vector();
	}

	void ComputerCpu::Read(array<double>^ result)
	{
		// 先頭ポインタ取得
		pin_ptr<double> resultPtr = &result[0];

		// 読み込み
		computer->Read(resultPtr);
	}
}}