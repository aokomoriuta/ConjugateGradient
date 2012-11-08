﻿using System;
using System.Threading.Tasks;
using LWisteria.Mgcg.LongVector;

namespace LWisteria.Mgcg
{
	/// <summary>
	/// マルチグリッド前処理付き共役勾配法のメインクラス
	/// </summary>
	static class MgcgMain
	{
		/// <summary>
		/// 未知数の数
		/// </summary>
		const int COUNT = 346578;

		/// <summary>
		/// 非ゼロ要素の最大数
		/// </summary>
		const int MAX_NONZERO_COUNT = 160;

		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		const int MIN_ITERATION = 0;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		const int MAX_ITERATION = COUNT;

		/// <summary>
		/// 収束誤差
		/// </summary>
		const double ALLOWABLE_RESIDUAL = 1e-4f;

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
			var cgGpuSingle = new ConjugateGradientSingleGpu(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);
			//var cgCLParallel = new ConjugateGradientParallelGpu(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);

			SparseMatrix A = new SparseMatrix(COUNT, MAX_NONZERO_COUNT);
			A.RowOffsets[0] = 0;
			for(int i = 0; i < COUNT; i++)
			{
				// この行の先頭位置を取得
				int rowOffset = A.RowOffsets[i];

				// 対角成分をゼロにする
				A.Elements[rowOffset] = 0;
				A.ColumnIndeces[rowOffset] = i;

				// 非ゼロ要素数
				int nonzeroCount = 1;

				// 各列で
				for(int j = (int)Math.Max(0, i - MAX_NONZERO_COUNT / 2 + 1); j < Math.Min(COUNT, i + MAX_NONZERO_COUNT / 2); j++)
				{
					if(i != j)
					{
						// 要素を計算
						var a_ij = (double)Math.Abs(Math.Sin(i + j));

						// 要素を設定
						A.Elements[rowOffset + nonzeroCount] = a_ij;
						A.ColumnIndeces[rowOffset + nonzeroCount] = j;
						nonzeroCount++;

						// 対角成分に追加
						A.Elements[rowOffset] += a_ij;
					}
				}

				A.RowOffsets[i + 1] = A.RowOffsets[i] + nonzeroCount;
			}

			cgCpu.A = A;
			cgGpuSingle.A = A;

			// 各行で
			Parallel.For(0, COUNT, (i) =>
			//for(int i = 0; i < COUNT - 1; i++)
			{
				// 右辺ベクトルを設定
				double b_i = (double)Math.Cos(i) * 10;
				cgCpu.b[i] = b_i;
				cgGpuSingle.b[i] = b_i;
				//cgCLParallel.b[i] = b_i;

				// 未知数を初期化
				double x_i = (double)i / 100;
				cgCpu.x[i] = x_i;
				cgGpuSingle.x[i] = x_i;
				//cgCLParallel.x[i] = x_i;
			}
			);

			// 開始を通知
			Console.WriteLine("start");

			// ストップウォッチを作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			//try
			{
				// CPUで方程式を解く
				stopwatch.Restart();
				cgCpu.Solve();
				stopwatch.Stop();
				var cpuTime = stopwatch.ElapsedTicks;

				// 1GPUで方程式を解く
				cgGpuSingle.Initialize();
				stopwatch.Restart();
				cgGpuSingle.Solve();
				stopwatch.Stop();
				cgGpuSingle.Read();
				var clSignleTime = stopwatch.ElapsedTicks;

				// 全要素の
				for(int i = 0; i < COUNT; i++)
				{
					// 誤差を取得
					double residual = Math.Abs(cgCpu.x[i] - cgGpuSingle.x[i]);

					// 許容誤差以上だったら
					if(residual / cgCpu.x[i] > 0.01)
					{
						// 通知
						Console.WriteLine("Single {0,4}: {1,5:e} (CPU:{2,5:e} vs GPU:{3,5:e})", i, residual / cgCpu.x[i] * 100, cgCpu.x[i], cgGpuSingle.x[i]);
					}
				}

				//// 複数GPUで方程式を解く
				//cgCLParallel.Initialize();
				//stopwatch.Restart();
				//cgCLParallel.Solve();
				//stopwatch.Stop();
				//var clParallelTime = stopwatch.ElapsedTicks;

				//// 全要素の
				//for(int i = 0; i < COUNT; i++)
				//{
				//	// 誤差を取得
				//	double residual = Math.Abs(cgCpu.x[i] - cgCLParallel.x[i]);

				//	// 許容誤差以上だったら
				//	if(residual > ALLOWABLE_RESIDUAL)
				//	{
				//		// 通知
				//		//Console.WriteLine("Parallel {0,4}: {1:e} ({2:e} vs {3:e})", i, residual, cgCpu.x[i], cgCLParallel.x[i]);
				//	}
				//}

				// かかった時間を表示
				Console.WriteLine("単一CPU: {0, 8} / {1} = {2, 8}", cpuTime, cgCpu.Iteration, cpuTime / System.Math.Max(1, cgCpu.Iteration));
				Console.WriteLine("単一GPU: {0, 8} / {1} = {2, 8}", clSignleTime, cgGpuSingle.Iteration, clSignleTime / System.Math.Max(1, cgGpuSingle.Iteration));
				//Console.WriteLine("複数GPU: {0} / {1} = {2}", clParallelTime, cgCLParallel.Iteration, clParallelTime / cgCLParallel.Iteration);
			}
			//catch(Exception ex)
			//{
			//    Console.WriteLine("!!!!{0}", ex.Message);
			//}

			// 終了
			Console.WriteLine("終了します");
			Console.ReadKey();
			return System.Environment.ExitCode;
		}
	}
}