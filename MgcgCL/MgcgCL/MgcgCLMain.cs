﻿using System;
using System.Threading.Tasks;
using LWisteria.MgcgCL.LongVector;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// OpenCLでマルチグリッド前処理付き共役勾配法のメインクラス
	/// </summary>
	static class MgcgCLMain
	{
		/// <summary>
		/// 未知数の数
		/// </summary>
		const int COUNT = 150000;

		/// <summary>
		/// 非ゼロ要素の最大数
		/// </summary>
		const int MAX_NONZERO_COUNT = 30;

		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		const int MIN_ITERATION = 0;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		const int MAX_ITERATION = 20000;

		/// <summary>
		/// 収束誤差
		/// </summary>
		const float ALLOWABLE_RESIDUAL = 1e-3f;

		/// <summary>
		/// エントリポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			// 要素数表示
			Console.WriteLine("N={0}", COUNT);

			// CG法を作成
			var cgCpu = new ConjugateGradientCpu(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);
			var cgCL = new ConjugateGradientCL(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);

			// 係数行列の初期化
			for(int i = 0; i < COUNT; i++)
			{
				cgCpu.A[i, i] = 0;

				cgCL.A[i, i] = 0;
			}

			// 各行で
			Parallel.For(0, COUNT, (i) =>
			//for(int i = 0; i < COUNT - 1; i++)
			{
				// 各列で
				for(int j = (int)Math.Max(0, i - MAX_NONZERO_COUNT / 2 + 1); j < Math.Min(COUNT, i + MAX_NONZERO_COUNT / 2); j++)
				{
					if(i != j)
					{
						// 要素を計算
						var a_ij = (float)Math.Abs(Math.Sin(i + j));

						// 要素を設定
						cgCpu.A[i, j] = a_ij;
						cgCL.A[i, j] = a_ij;

						// 対角成分に追加
						cgCpu.A[i, i] += a_ij;
						cgCL.A[i, i] += a_ij;
					}
				}

				// 生成項を設定
				float b_i = i * 0.01f;
				cgCpu.b[i] = b_i;
				cgCL.b[i] = b_i;

				// 未知数を初期化
				float x_i = 0;
				cgCpu.x[i] = x_i;
				cgCL.x[i] = x_i;
			}
			);

			//for(int i = 0; i < COUNT; i++)
			//{
			//    for(int j = 0; j < COUNT; j++)
			//    {
			//        Console.Write("{0,5:f} ", cgCpu.A[i, j]);
			//    }


			//    Console.WriteLine("");
			//}


			//// 固有値を計算
			//var eigenValues = new System.Collections.Generic.List<float>(cgCpu.A.GetEigenValues(MAX_ITERATION, ALLOWABLE_RESIDUAL));

			//eigenValues.Sort((Comparison<float>)((x, y) => (x == y) ? 0 : (x < y) ? 1 : -1));

			//// 固有値を表示
			//foreach(var eigenValue in eigenValues)
			//{
			//    Console.WriteLine(eigenValue);
			//}

			// 開始を通知
			Console.WriteLine("start");

			// ストップウォッチを作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			try
			{

				// CPUで方程式を解く
				stopwatch.Restart();
				cgCpu.Solve();
				stopwatch.Stop();
				var cpuTime = stopwatch.ElapsedMilliseconds;

				// OpenCLで方程式を解く
				stopwatch.Restart();
				cgCL.Solve();
				stopwatch.Stop();
				var clTime = stopwatch.ElapsedMilliseconds;

				for(int i = 0; i < COUNT; i++)
				{
					float residual = Math.Abs(cgCpu.x[i] - cgCL.x[i]);

					if(residual > ALLOWABLE_RESIDUAL)
					{
						Console.WriteLine("{0,4}: {1:e}", i, residual);
					}
				}

				// かかった時間を表示
				Console.WriteLine("CPU: {0} / {1} = {2}", cpuTime, cgCpu.Iteration, cpuTime / cgCpu.Iteration);
				Console.WriteLine(" CL: {0} / {1} = {2}", clTime, cgCL.Iteration, clTime / cgCL.Iteration);
			}
			catch(Exception ex)
			{
				Console.WriteLine("!!!!{0}", ex.Message);
			}

			// 終了
			Console.WriteLine("終了します");
			Console.ReadKey();
			return System.Environment.ExitCode;
		}
	}
}