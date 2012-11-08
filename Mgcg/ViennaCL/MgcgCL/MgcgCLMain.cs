using System;
using System.Collections.Generic;

namespace LWisteria.StudiesOfOpenCL.SimpleViennaCL
{
	/// <summary>
	/// 共役勾配法プログラム
	/// </summary>
	static class MgcgCLMain
	{
		/// <summary>
		/// 要素数
		/// </summary>
		const int N = 10;

		/// <summary>
		/// バンド幅
		/// </summary>
		const int BAND_WIDTH = 6;

		/// <summary>
		/// 繰り返し回数
		/// </summary>
		const int ITRATION = 1;

		const double RESIDUAL = (double)1e-4;

		const int MIN_ITERATION = 0;

		const int MAX_ITERATION = N;

		/// <summary>
		/// エントリポイント
		/// </summary>
		/// <returns></returns>
		static int Main()
		{
			// 配列を初期化
			var x = new double[N];
			var A = new CompuressedMatrix();
			var b = new double[N];

			// 各要素の値を
			for(int i = 0; i < N; i++)
			{
				A[i, i] = i;

                for (int j = System.Math.Max(0, i - BAND_WIDTH / 2); j <= System.Math.Min(N - 1, i + BAND_WIDTH / 2); j++)
                {
                    if (i != j)
                    {
                        double a_ij = (double)System.Math.Abs(System.Math.Sin(i + j));
                        A[i, j] = a_ij;

                        A[i, i] += a_ij;
                    }
                }

				// それぞれ設定
				x[i] = 0;
				b[i] = 1 + 0.1* i;// (double)System.Math.Asin((double)(i + 1) / N);
			}
			Console.WriteLine("{0}x{1}", N, BAND_WIDTH);

			/*
			// 条件表示
			for(int i = 0; i < N; i++)
			{
				Console.Write("| ");
				for(int j = 0; j < N; j++)
				{
					Console.Write("{0:0.00}, ", A[i, j]);
				}

				Console.WriteLine("|");
			}
			Console.WriteLine("x");
			for(int i = 0; i < N; i++)
			{
				Console.WriteLine("|{0:0.00}|", b[i]);
			}
			Console.WriteLine("=");
			*/

			var elementsList = new List<double>();
			var columnIndecesList = new List<uint>();
			var rowOffsets = new uint[N + 1];
			rowOffsets[0] = 0;
			for(int i = 0; i < N; i++)
			{
				columnIndecesList.AddRange(A.Elements[i].Keys);
				elementsList.AddRange(A.Elements[i].Values);

				rowOffsets[i + 1] = rowOffsets[i] + (uint)A.Elements[i].Count;
			}
			var elements = elementsList.ToArray();
			var columnIndeces = columnIndecesList.ToArray();

			// ストップウォッチ作成
			var stopwatch = new System.Diagnostics.Stopwatch();

            Console.WriteLine("CPU => ");
            for (int iteration = 0; iteration < ITRATION; iteration++)
			{
				// 結果を初期化
				for(int i = 0; i < N; i++)
				{
					x[i] = 0;
				}

				// CPUで加算を作成
				var computer = new LWisteria.Mgcg.ComputerCpu(N);

				Console.WriteLine("Time = ");
				// 入力
				stopwatch.Restart();
				computer.Write(elements, rowOffsets, columnIndeces, x, b);
				//computer.Write(A.Elements, x, b);
				var inputTime = stopwatch.ElapsedMilliseconds;

				// 演算実行
				stopwatch.Restart();
				{
					computer.Solve(RESIDUAL, MIN_ITERATION, MAX_ITERATION);
				}
				var executionTime = stopwatch.ElapsedMilliseconds;

				// 出力
				stopwatch.Restart();
				computer.Read(x);
				var outputTime = stopwatch.ElapsedMilliseconds;

				// 実行時間表示
				Console.WriteLine("{0, 5} + {1, 5} + {2, 5} = {3, 5} / {4}", inputTime, executionTime, outputTime, inputTime + executionTime + outputTime, computer.Iteration());

				//*
				// 結果表示
				for(int i = 0; i < N; i++)
				{
					Console.WriteLine("|{0:0.00}|", x[i]);
				}
				//*/
			}


            Console.WriteLine("GPU => ");
            for (int iteration = 0; iteration < ITRATION; iteration++)
			{
				// 結果を初期化
				for(int i = 0; i < N; i++)
				{
					x[i] = 0;
				}

				// GPUで加算を作成
				var computer = new LWisteria.Mgcg.ComputerGpu(N);

				Console.WriteLine("Time = ");

				// 入力
				stopwatch.Restart();
				computer.Write(elements, rowOffsets, columnIndeces, x, b);
				var inputTime = stopwatch.ElapsedMilliseconds;

				// 演算実行
				stopwatch.Restart();
				{
					computer.Solve(RESIDUAL, MIN_ITERATION, MAX_ITERATION);
				}
				var executionTime = stopwatch.ElapsedMilliseconds;

				// 出力
				stopwatch.Restart();
				computer.Read(x);
				var outputTime = stopwatch.ElapsedMilliseconds;

				// 実行時間表示

				Console.WriteLine("{0, 5} + {1, 5} + {2, 5} = {3, 5} / {4}", inputTime, executionTime, outputTime, inputTime + executionTime + outputTime, computer.Iteration());
				
				//*
				// 結果表示
				for(int i = 0; i < N; i++)
				{
					Console.WriteLine("|{0:0.00}|", x[i]);
				}
				//*/
			}

			return Environment.ExitCode;
		}
	}
}