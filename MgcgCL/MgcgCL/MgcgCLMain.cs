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
		const long COUNT = 1500000;

		/// <summary>
		/// 非ゼロ要素の最大数
		/// </summary>
		const int MAX_NONZERO_COUNT = 15;

		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		const int MIN_ITERATION = 0;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		const int MAX_ITERATION = (int)(COUNT/100);

		/// <summary>
		/// 収束誤差
		/// </summary>
		const double ALLOWABLE_RESIDUAL = 1e-8;

		/// <summary>
		/// エントリポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			Console.WriteLine("N={0}", COUNT);

			// CG法を作成
			var cgCpu = new ConjugateGradientCpu(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);
			var cgCL = new ConjugateGradientCL(COUNT, MAX_NONZERO_COUNT, MIN_ITERATION, MAX_ITERATION, ALLOWABLE_RESIDUAL);

			// 係数行列の初期化
			for(long i = 0; i < COUNT; i++)
			{
				cgCpu.A[i, i] = 0;
				cgCpu.isEnabled[i] = true;

				cgCL.A[i, i] = 0;
				cgCL.isEnabled[i] = true;
			}

			// 各行で
			Parallel.For(0, COUNT - 1, (i) =>
			//for(long i = 0; i < count - 1; i++)
			{
				// 各列で
				for(long j = i + 1; j < Math.Min(COUNT, i + MAX_NONZERO_COUNT / 2); j++)
				{
					// 要素を計算
					var a_ij = Math.Abs(Math.Sin(i * i + 2 * j));

					// 要素を設定
					cgCpu.A[i, j] = a_ij;
					cgCpu.A[j, i] = a_ij;
					cgCL.A[i, j] = a_ij;
					cgCL.A[j, i] = a_ij;

					// 対角成分を追加
					cgCpu.A[i, i] += a_ij;
					cgCpu.A[j, j] += a_ij;
					cgCL.A[i, i] += a_ij;
					cgCL.A[j, j] += a_ij;
				}

				// 生成項を設定
				double b_i = i * 0.01;
				cgCpu.b[i] = b_i;
				cgCL.b[i] = b_i;
			}
			);

			Console.WriteLine("start");

			// ストップウォッチを作成
			var stopwatch = new System.Diagnostics.Stopwatch();


			// CPUで方程式を解く
			stopwatch.Restart();
			cgCpu.Solve();
			stopwatch.Stop();
			var cpuTime = stopwatch.ElapsedMilliseconds;

			// OpenCLで方程式を説く
			stopwatch.Restart();
			cgCL.Solve();
			stopwatch.Stop();
			var clTime = stopwatch.ElapsedMilliseconds;

			// かかった時間を表示
			Console.WriteLine("CPU: {0} / {1} = {2}", cpuTime, cgCpu.Iteration, cpuTime/cgCpu.Iteration);
			Console.WriteLine(" CL: {0} / {1} = {2}", clTime, cgCL.Iteration, clTime/cgCL.Iteration);

			// 終了
            Console.WriteLine("終了します");
			Console.ReadKey();
			return System.Environment.ExitCode;
		}
	}
}