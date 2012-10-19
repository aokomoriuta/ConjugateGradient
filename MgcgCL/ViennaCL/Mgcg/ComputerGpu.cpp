#include "ComputerGpu.hpp"
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>

namespace LWisteria{ namespace Mgcg
{
#pragma unmanaged
	ComputerGpuNative::ComputerGpuNative(int n)
	{
		// 要素数を設定
		count = n;

		// GPU側メモリを確保
		x = std::shared_ptr<viennacl::vector<double>>(new viennacl::vector<double>(count));
		A = std::shared_ptr<viennacl::compressed_matrix<double>>(new viennacl::compressed_matrix<double>(count, count));
		b = std::shared_ptr<viennacl::vector<double>>(new viennacl::vector<double>(count));
		r = std::shared_ptr<viennacl::vector<double>>(new viennacl::vector<double>(count));
		p = std::shared_ptr<viennacl::vector<double>>(new viennacl::vector<double>(count));
		Ap = std::shared_ptr<viennacl::vector<double>>(new viennacl::vector<double>(count));
	}

	void ComputerGpuNative::SetMatrix(double elementsPtr[], unsigned int rowOffsetsPtr[], unsigned int columnIndecesPtr[], const int& elementsCount)
	{
		A->set(rowOffsetsPtr, columnIndecesPtr, elementsPtr, count, count, elementsCount);
	}

	void ComputerGpuNative::SetVector(const double xPtr[], const double bPtr[])
	{
		// 入力配列を複製
		viennacl::fast_copy(xPtr, xPtr+count, x->begin());
		viennacl::fast_copy(bPtr, bPtr+count, b->begin());

		// ここまで待機
		viennacl::ocl::get_queue().finish();
	}

	void ComputerGpuNative::Solve(double residual, int minIteration, int maxIteration)
	{
		using namespace viennacl::linalg;

		{
			// 初期値を設定
			/*
			* (Ap)_0 = A * x
			* r_0 = b - Ap
			* rr_0 = r_0・r_0
			* p_0 = (LDLr)_0
			*/
			*Ap = prod(*A, *x);
			*r = *b - *Ap;
			double rr = inner_prod(*r, *r);
			double rr0 = rr;
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
				double alpha = rr / inner_prod(*p, *Ap);
				*x += alpha * *p;
				*r -= alpha * *Ap;
				double rrNew = inner_prod(*r, *r);

				//std::cout << iteration << ": " << sqrt(rrNew) << std::endl;

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
					double beta = rrNew / rr;
					*p = *r + beta * *p;

					rr = rrNew;
				}
			}
		}

		/*
		auto pre = jacobi_precond<viennacl::compressed_matrix<double>>(*A, jacobi_tag());
		auto cgtag = cg_tag(residual);
		*x = viennacl::linalg::solve(*A, *b, cgtag, pre);
		iteration = cgtag.iters();
		*/

		// ここまで待機
		viennacl::ocl::get_queue().finish();
	}

	void ComputerGpuNative::Read(double xPtr[])
	{
		// 結果を複製
		viennacl::fast_copy(x->begin(), x->end(), xPtr);

		// ここまで待機
		viennacl::ocl::get_queue().finish();
	}

	int ComputerGpuNative::Iteration()
	{
		return iteration;
	}

#pragma managed
	ComputerGpu::ComputerGpu(int n)
	{
		// 計算クラスを作成
		computer = new ComputerGpuNative(n);
	}
	
	ComputerGpu::~ComputerGpu()
	{
		// 計算クラスを廃棄
		delete(computer);
	}
	
	void ComputerGpu::Write(array<double>^ elements, array<unsigned int>^ rowOffsets, array<unsigned int>^ columnIndeces, array<double>^ x, array<double>^ b)
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

	void ComputerGpu::Solve(double residual, int minIteration, int maxIteration)
	{
		// 演算実行
		computer->Solve(residual, minIteration, maxIteration);
	}

	void ComputerGpu::Read(array<double>^ x)
	{
		// 先頭ポインタ取得
		pin_ptr<double> xPtr = &x[0];

		// 読み込み
		computer->Read(xPtr);
	}

	int ComputerGpu::Iteration()
	{
		return computer->Iteration();
	}

}}