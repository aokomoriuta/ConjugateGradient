#pragma once
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include <memory>

namespace LWisteria{ namespace Mgcg
{
	//! GPUで計算するクラス
	class ComputerGpuNative
	{
	private:
		//! 要素数
		int count;

		//! 未知数ベクトル
		std::shared_ptr<viennacl::vector<double>> x;

		//! 係数行列
		std::shared_ptr<viennacl::compressed_matrix<double>> A;

		//! 右辺ベクトル
		std::shared_ptr<viennacl::vector<double>> b;

		//! 残差ベクトル
		std::shared_ptr<viennacl::vector<double>> r;

		//! 探索方向ベクトル
		std::shared_ptr<viennacl::vector<double>> p;

		//! 係数行列と探索方向ベクトルの積
		std::shared_ptr<viennacl::vector<double>> Ap;

		//! 繰り返し回数
		int iteration;

	public:
		/* GPUで計算するクラスを生成する
		* @param n 要素数
		*/
		ComputerGpuNative(int n);

		/** 行列の要素を設定する
		* @param i 行番号
		* @param j 列番号
		* @param value 要素値
		*/
		void SetMatrix(double elementsPtr[], unsigned int rowOffsetsPtr[], unsigned int columnIndecesPtr[], const int& elementsCount);

		/** @brief ベクトルを設定する
		* @param vectorPtr ベクトルの先頭ポインタ
		*/
		void SetVector(const double xPtr[], const double bPtr[]);

		//! 演算を実行する
		void Solve(double residual, int minIteration, int maxIteration);

		/** @brief 出力値を読み込む
		* @param resultPtr 結果を格納するベクトル
		*/
		void Read(double xPtr[]);

		/** @brief 繰り返し回数を取得する
		*/
		int Iteration();
	};

	//! GPUで計算するクラスのラッパー
	public ref class ComputerGpu
	{
	private:
		//! ネイティブ側の計算クラス
		ComputerGpuNative* computer;

	public:
		/* GPUで計算するクラスを生成する
		* @param n 要素数
		*/
		ComputerGpu(int n);

		//! CPUで計算するクラスを破棄する
		~ComputerGpu();

		/* 入力値を書き込む
		* @param left 足されるベクトルの配列
		* @param right 足すベクトルの配列
		*/
		void Write(array<double>^ elements, array<unsigned int>^ rowOffsets, array<unsigned int>^ columnIndeces, array<double>^ x, array<double>^ b);
		
		//! 演算を実行する
		void Solve(double residual, int minIteration, int maxIteration);

		/* 出力値を読み込む
		* @param result 結果を格納する配列
		*/
		void Read(array<double>^ x);

		/** @brief 繰り返し回数を取得する
		*/
		int Iteration();
	};
}}