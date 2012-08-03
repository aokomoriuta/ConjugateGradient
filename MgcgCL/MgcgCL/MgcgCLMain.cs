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
		const int count = 10;

		/// <summary>
		/// 非ゼロ要素の最大数
		/// </summary>
		const int maxNonzeroCount = count;

		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		const int minIteration = count / 10;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		const int maxiteration = count;

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
			// CPUでのCG法を作成
			var cg = new ConjugateGradient(count, maxNonzeroCount, minIteration, maxNonzeroCount, allowableResidual);

			// 係数行列の初期化
			for(int i = 0; i < count; i++)
			{
				cg.A[i, i] = 0;
				cg.isEnabled[i] = true;
			}

			// 各行で
			for(int i = 0; i < count-1; i++)
			{
				// 各列で
				for(int j = i+1; j < count; j++)
				{
					// 要素を計算
					var a_ij = Math.Abs(Math.Sin(i*i + 2*j));

					// 要素を設定
					cg.A[i, j] = a_ij;
					cg.A[j, i] = a_ij;

					// 対角成分を追加
					cg.A[i, i] += a_ij;
					cg.A[j, j] += a_ij;
				}

				// 生成項を設定
				cg.b[i] = i * 0.2;
			}

			// 各行で
			for(int i = 0; i < count; i++)
			{
				// 各列で
				for(int j = 0; j < count; j++)
				{
					// 行列の要素を表示
					Console.Write("{0,5:f} ", cg.A[i, j]);
				}

				// 生成項の要素を表示
				Console.WriteLine("  {0,5:f}", cg.b[i]);
			}

			// OpenCLで解く
			//cg.SolveCL();


			// 変数ベクトルを
			for(int i = 0; i < count; i++)
			{
				// 初期化
				cg.x[i] = 0;
			}

			// 方程式を解く
			cg.Solve();

			// 解の全てを
			foreach(var x in cg.x)
			{
				// 出力
				Console.WriteLine("{0}", x);
			}

			// 終了
			Console.ReadKey();
			return System.Environment.ExitCode;
		}
	}
}