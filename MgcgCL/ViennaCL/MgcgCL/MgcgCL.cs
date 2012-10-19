using System;
using System.Collections.Generic;

namespace LWisteria.StudiesOfOpenCL.SimpleViennaCL
{
	/// <summary>
	/// ViennaCLを使う試験を兼ねた簡易プログラム
	/// </summary>
	static class SimpleViennaCLMain
	{
		/// <summary>
		/// 要素数
		/// </summary>
		const int N = 34567*3;

		/// <summary>
		/// バンド幅
		/// </summary>
		const int BAND_WIDTH = 160;

		/// <summary>
		/// 繰り返し回数
		/// </summary>
		const int ITRATION = 1;

		const double RESIDUAL = (double)1e-8;

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

				for(int j = System.Math.Max(0, i - BAND_WIDTH / 2); j <= System.Math.Min(N - 1, i + BAND_WIDTH / 2); j++)
				{
					if(i != j)
					{
						double a_ij = (double)System.Math.Abs(System.Math.Sin(i + j));
						A[i, j] = a_ij;

						A[i, i] += a_ij;
					}
				}

				// それぞれ設定
				x[i] = 0;
				b[i] = (double)System.Math.Asin((double)i / N);
			}
			Console.WriteLine("N: {0}", N);

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

			// ストップウォッチ作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			Console.WriteLine("CPU => ");
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
				computer.Write(A.Elements, x, b);
				var inputTime = stopwatch.ElapsedMilliseconds;

				// 演算実行
				stopwatch.Restart();
				for(int i = 0; i < ITRATION; i++)
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

				/*
				// 結果表示
				for(int i = 0; i < N; i++)
				{
					Console.WriteLine("|{0:0.00}|", x[i]);
				}
				//*/
			}

			Console.WriteLine("GPU => ");
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

				/*
				System.Collections.Generic.IEnumerable<System.Collections.Generic.KeyValuePair<int, double>> rows = new System.Collections.Generic.KeyValuePair<int, double>[0];
				A.Elements.ForEach(row => { rows = rows.Concat(row); });
				var columnIndeces = rows.Select(row => (uint)row.Key).ToArray();
				var elements = rows.Select(row => row.Value).ToArray();
				var rowOffsets = new uint[] { 0 }.Concat(A.Elements.Select((row, i) =>
				{
					uint offset = 0;

					for(int j = 0; j < i + 1; j++)
					{
						offset += (uint)A.Elements[j].Count;
					}

					return offset;
				}).ToArray()).ToArray();
				*/
				var elements = new List<double>();
				var columnIndeces = new List<uint>();
				var rowOffsets = new uint[N + 1];
				rowOffsets[0] = 0;
				for(int i = 0; i < N; i++)
				{
					columnIndeces.AddRange(A.Elements[i].Keys);
					elements.AddRange(A.Elements[i].Values);

					rowOffsets[i + 1] = rowOffsets[i] + (uint)A.Elements[i].Count;
				}

				computer.Write(elements.ToArray(), rowOffsets, columnIndeces.ToArray(), x, b);
				var inputTime = stopwatch.ElapsedMilliseconds;

				// 演算実行
				stopwatch.Restart();
				for(int i = 0; i < ITRATION; i++)
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
				/*
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