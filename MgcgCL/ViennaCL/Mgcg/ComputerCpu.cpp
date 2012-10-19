#include "ComputerCpu.hpp"

namespace LWisteria{ namespace Mgcg
{
#pragma unmanaged
typedef boost::numeric::ublas::vector<double> Vector;
typedef boost::numeric::ublas::compressed_matrix<double> SparseMatrix;

	ComputerCpuNative::ComputerCpuNative(const int& n)
	{
		// 要素数を設定
		count = n;

		// ベクトルと行列を生成
		x = std::shared_ptr<Vector>(new Vector(count));
		A = std::shared_ptr<SparseMatrix>(new SparseMatrix(count, count, 44));
		b = std::shared_ptr<Vector>(new Vector(count));
		r = std::shared_ptr<Vector>(new Vector(count));
		p = std::shared_ptr<Vector>(new Vector(count));
		Ap = std::shared_ptr<Vector>(new Vector(count));
	}

	void ComputerCpuNative::SetMatrix(double elementsPtr[], unsigned int rowOffsetsPtr[], unsigned int columnIndecesPtr[], const int& elementsCount)
	{
		A->reserve(elementsCount, false);

		// データを複製
		std::copy(elementsPtr, elementsPtr + elementsCount, A->value_data().begin());
		std::copy(rowOffsetsPtr, rowOffsetsPtr + count + 1, A->index1_data().begin());
		std::copy(columnIndecesPtr, columnIndecesPtr + elementsCount, A->index2_data().begin());
		
		A->set_filled(count+1, elementsCount);
	}

	void ComputerCpuNative::SetVector(const double xPtr[], const double bPtr[])
	{
		// 入力配列を複製
		std::copy(xPtr, xPtr+count, x->begin());
		std::copy(bPtr, bPtr+count, b->begin());
	}

	void ComputerCpuNative::Solve(double residual, int minIteration, int maxIteration)
	{
		using namespace boost::numeric::ublas;

		// 初期値を設定
		/*
		* (Ap)_0 = A * x
		* r_0 = b - Ap
		* rr_0 = r_0・r_0
		* p_0 = r_0
		*/
		*Ap = prod(*A, *x);
		*r = *b - *Ap;
		auto rr = inner_prod(*r, *r);
		auto rr0 = rr;
		*p = *r;

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
			*Ap = prod(*A, *p);
			auto alpha = rr / inner_prod(*p, *Ap);
			*x += alpha * *p;
			*r -= alpha * *Ap;
			auto rrNew = inner_prod(*r, *r);

			//std::cout << iteration << ": " << sqrt(rrNew/rr0) << std::endl;

			// 収束したかどうかを取得
			converged = (minIteration < iteration) && (rrNew/rr0  < residual * residual);

			// 収束していなかったら
			if(!converged)
			{
				// 残りの計算を実行
				/*
				* β= r'r'/rLDLr
				* p = r' + βp
				*/
			auto beta = rrNew / rr;
			*p = *r + beta * *p;

			rr = rrNew;
			}
		}
	}

	void ComputerCpuNative::Read(double xPtr[])
	{
		// 結果を複製
		std::copy(x->begin(), x->end(), xPtr);
	}

	int ComputerCpuNative::Iteration()
	{
		return iteration;
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
	
	void ComputerCpu::Write(array<double>^ elements, array<unsigned int>^ rowOffsets, array<unsigned int>^ columnIndeces, array<double>^ x, array<double>^ b)
	{
		// 行列を設定
		pin_ptr<double> elementsPtr = &elements[0];
		pin_ptr<unsigned int> rowOffsetsPtr = &rowOffsets[0];
		pin_ptr<unsigned int> columnIndecesPtr = &columnIndeces[0];
		int elementsCount = elements->Length;
		computer->SetMatrix(elementsPtr, rowOffsetsPtr, columnIndecesPtr, elementsCount);

		// ベクトルを設定
		pin_ptr<double> xPtr = &x[0];
		pin_ptr<double> bPtr = &b[0];
		computer->SetVector(xPtr, bPtr);
	}

	void ComputerCpu::Solve(double residual, int minIteration, int maxIteration)
	{
		// 演算実行
		computer->Solve(residual, minIteration ,maxIteration);
	}

	void ComputerCpu::Read(array<double>^ x)
	{
		// 先頭ポインタ取得
		pin_ptr<double> xPtr = &x[0];

		// 読み込み
		computer->Read(xPtr);
	}

	int ComputerCpu::Iteration()
	{
		return computer->Iteration();
	}
}}