#pragma once
#include <iostream>
#include <memory>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#using<System.dll>

namespace LWisteria{ namespace Mgcg
{
	//! CPUで計算するクラス
	class ComputerCpuNative
	{
	private:
		//! 要素数
		int count;
		
		//! 未知ベクトル
		std::shared_ptr<boost::numeric::ublas::vector<double>> x;

		//! 係数行列
		std::shared_ptr<boost::numeric::ublas::compressed_matrix<double>> A;

		//! 右辺ベクトル
		std::shared_ptr<boost::numeric::ublas::vector<double>> b;

		//! 残差ベクトル
		std::shared_ptr<boost::numeric::ublas::vector<double>> r;

		//! 探索方向ベクトル
		std::shared_ptr<boost::numeric::ublas::vector<double>> p;

		//! 係数行列と探索方向ベクトルの積
		std::shared_ptr<boost::numeric::ublas::vector<double>> Ap;

		//! 繰り返し回数
		int iteration;

	public:
		/* CPUで計算するクラスを生成する
		* @param n 要素数
		*/
		ComputerCpuNative(const int& n);

		/** 係数行列の要素を設定する
		* @param i 行番号
		* @param j 列番号
		* @param value 要素値
		*/
		void SetMatrix(double elementsPtr[], unsigned int rowOffsetsPtr[], unsigned int columnIndecesPtr[], const int& elementsCount);

		/** @brief 右辺ベクトルを設定する
		* @param vectorPtr ベクトルの先頭ポインタ
		*/
		void SetVector(const double xPtr[], const double bPtr[]);

		//! 演算を実行する
		void Solve(double residual, int minIteration, int maxIteration);

		/* 出力値を読み込む
		* @param resultPtr 結果を格納するベクトル
		*/
		void Read(double xPtr[]);

		/** @brief 繰り返し回数を取得する
		*/
		int Iteration();
	};

	//! CPUで計算するクラスのラッパー
	public ref class ComputerCpu
	{
	private:
		//! ネイティブ側の計算クラス
		ComputerCpuNative* computer;

	public:
		/* CPUで計算するクラスを生成する
		* @param n 要素数
		*/
		ComputerCpu(int n);

		//! CPUで計算するクラスを破棄する
		~ComputerCpu();

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