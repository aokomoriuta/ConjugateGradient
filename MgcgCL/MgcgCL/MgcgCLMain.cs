using System;
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
		const int COUNT = 345678;//1234;

		/// <summary>
		/// 非ゼロ要素の最大数
		/// </summary>
		const int MAX_NONZERO_COUNT = 160;

		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		const int MIN_ITERATION = 50;

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
			var cgCLSingle = new ConjugateGradientCLSingle(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);
			var cgCLParallel = new ConjugateGradientCLParallel(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);

			// 係数行列の初期化
			for(int i = 0; i < COUNT; i++)
			{
				cgCpu.A[i, i] = 0;
				cgCLSingle.A[i, i] = 0;
				cgCLParallel.A[i, i] = 0;
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
						var a_ij = (double)Math.Abs(Math.Sin(i + j));

						// 要素を設定
						cgCpu.A[i, j] = a_ij;
						cgCLSingle.A[i, j] = a_ij;
						cgCLParallel.A[i, j] = a_ij;

						// 対角成分に追加
						cgCpu.A[i, i] += a_ij;
						cgCLSingle.A[i, i] += a_ij;
						cgCLParallel.A[i, i] += a_ij;
					}
				}

				// 右辺ベクトルを設定
				double b_i = (double)Math.Cos(i) * 10;
				cgCpu.b[i] = b_i;
				cgCLSingle.b[i] = b_i;
				cgCLParallel.b[i] = b_i;

				// 未知数を初期化
				double x_i = (double)i / 100;
				cgCpu.x[i] = x_i;
				cgCLSingle.x[i] = x_i;
				cgCLParallel.x[i] = x_i;
			}
			);

			// 開始を通知
			Console.WriteLine("start");
			Console.ReadKey();

			// ストップウォッチを作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			try
			{
				// CPUで方程式を解く
				stopwatch.Restart();
				cgCpu.Solve();
				stopwatch.Stop();
				var cpuTime = stopwatch.ElapsedTicks;

				// 1GPUで方程式を解く
				cgCLSingle.Initialize();
				stopwatch.Restart();
				cgCLSingle.Solve();
				stopwatch.Stop();
				cgCLSingle.Read();
				var clSignleTime = stopwatch.ElapsedTicks;

				// 全要素の
				for(int i = 0; i < COUNT; i++)
				{
					// 誤差を取得
					double residual = Math.Abs(cgCpu.x[i] - cgCLSingle.x[i]);

					// 許容誤差以上だったら
					if(residual > ALLOWABLE_RESIDUAL)
					{
						// 通知
						Console.WriteLine("Single {0,4}: {1:e} (CPU{2:e} vs GPU{3:e})", i, residual, cgCpu.x[i], cgCLSingle.x[i]);
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
				Console.WriteLine("単一CPU: {0} / {1} = {2}", cpuTime, cgCpu.Iteration, cpuTime / cgCpu.Iteration);
				Console.WriteLine("単一GPU: {0} / {1} = {2}", clSignleTime, cgCLSingle.Iteration, clSignleTime / cgCLSingle.Iteration);
				//Console.WriteLine("複数GPU: {0} / {1} = {2}", clParallelTime, cgCLParallel.Iteration, clParallelTime / cgCLParallel.Iteration);
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