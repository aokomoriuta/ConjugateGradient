using System;
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
		const int count = 10000;

		/// <summary>
		/// 非ゼロ要素の最大数
		/// </summary>
		const int maxNonzeroCount = 15*2;

		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		const int minIteration = 0;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		const int maxIteration = count;

		/// <summary>
		/// 収束誤差
		/// </summary>
		const double allowableResidual = 1e-8;

		/// <summary>
		/// エントリポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			// CG法を作成
			var cgCpu = new ConjugateGradient(count, maxNonzeroCount, minIteration, maxIteration, allowableResidual);
			var cgCL = new ConjugateGradientCL(count, maxNonzeroCount, minIteration, maxIteration, allowableResidual);

			// 係数行列の初期化
			for(int i = 0; i < count; i++)
			{
				cgCpu.A[i, i] = 0;
				cgCpu.isEnabled[i] = true;

				cgCL.A[i, i] = 0;
				cgCL.isEnabled[i] = true;
			}

			// 各行で
			for(int i = 0; i < count-1; i++)
			{
				// 各列で
				for(int j = i+1; j < count; j++)
				{
					if ((i-j)*(i-j) < maxNonzeroCount*maxNonzeroCount/4)
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
				}
				
				// 生成項を設定
				double b_i = Math.Cos(i * i * 0.1);
				cgCpu.b[i] = b_i;
				cgCL.b[i] = b_i;
			}

			// 各行で
			for (int i = 0; i < count; i++)
			{
				// 各列で
				for (int j = 0; j < count; j++)
				{
					// 行列の要素を表示
					//Console.Write("{0,5:f2} ", cgCpu.A[i, j]);
				}

				// 生成項の要素を表示
				//Console.WriteLine("  {0,5:f2}", cgCpu.b[i]);
			}

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

			// 解の全てを
			for(int i = 0; i < count; i++)
			{
				// 精度以下切り捨て
				cgCpu.x[i] = Math.Round(cgCpu.x[i] / allowableResidual) * allowableResidual;
				cgCL.x[i] = Math.Round(cgCL.x[i] / allowableResidual) * allowableResidual;

				// 答えが違ったら
				if(cgCpu.x[i] != cgCL.x[i])
				{
					// 出力
					Console.WriteLine("{0}: {0} vs {1}", i, cgCpu.x[i], cgCL.x[i]);
				}
			}

			// かかった時間を表示
			Console.WriteLine("CPU: {0}", cpuTime);
			Console.WriteLine(" CL: {0}", clTime);

			// 終了
            Console.WriteLine("終了します");
			Console.ReadKey();
			return System.Environment.ExitCode;
		}
	}
}