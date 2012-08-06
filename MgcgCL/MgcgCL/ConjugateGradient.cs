using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 共役勾配法
	/// </summary>
	public class ConjugateGradient : LinerEquations
	{
		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		int minIteration;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		int maxIteration;

		/// <summary>
		/// 収束判定誤差の2乗
		/// </summary>
		double allowableResidual2;

		/// <summary>
		///  残差ベクトル
		/// </summary>
		readonly double[] r;

		/// <summary>
		/// 探索方向ベクトル
		/// </summary>
		readonly double[] p;

		/// <summary>
		/// 係数行列と探索方向ベクトルの積
		/// </summary>
		readonly double[] Ap;

		/// <summary>
		/// 共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradient(long count, long maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount)
		{
			// 残差および探索方向と係数行列と探索方向の積を初期化
			this.r = new double[count];
			this.p = new double[count];
			this.Ap = new double[count];

			// 最小・最大繰り返し回数を設定
			this.minIteration = _minIteration;
			this.maxIteration = _maxIteration;

			// 収束判定誤差を設定
			this.allowableResidual2 = allowableResidual * allowableResidual;
		}

		/// <summary>
		/// 方程式を解く
		/// </summary>
		public void Solve()
		{
			// ベクトルをゼロに初期化
			for(long i = 0; i < this.Count; i++)
			{
				this.Ap[i] = 0;
				this.r[i] = 0;
			}

			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = (LDLr)_0
			 */
			this.A.Multiply(this.Ap, this.x, this.isEnabled);
			this.r.SetAdded(this.b, this.Ap, -1.0);
			this.r.CopyTo(this.p, 0);

			// 収束したかどうか
			bool converged = false;

			// 収束しない間繰り返す
			for(int iteration = 0; !converged; iteration++)
			{
				// 計算を実行
				/*
				 * rr = r・r
				 * Ap = A * p
				 * α = rr/(p・Ap)
				 * x' += αp
				 * r' -= αAp
				 * r'r' = r'・r'
				 */
				double rr = this.r.Dot(this.r);
				this.A.Multiply(this.Ap, this.p, this.isEnabled);
				double alpha = rr / this.p.Dot(this.Ap);
				this.x.SetAdded(this.x, this.p, alpha);
				this.r.SetAdded(this.r, this.Ap, -alpha);
				double rrNew = this.r.Dot(this.r);

				Console.WriteLine("{0}: {1}", iteration, this.Ap.Dot(this.Ap));

				// 最小繰り返し回数未満なら
				if(iteration < this.minIteration)
				{
					// 収束していない
					converged = false;
				}
				// 最大繰り返し回数を超えていたら
				else if(iteration > this.maxIteration)
				{
					// 例外
					throw new System.ApplicationException("圧力方程式が収束しませんでした。");
				}
				// それ以外の時
				else
				{
					// 残差ベクトルの大きさが収束判定誤差より小さいかどうかを計算
					converged = (rrNew < this.allowableResidual2);
				}

				// 収束していたら
				if(converged)
				{
					// 計算終了
					break;
				}

				// 残りの計算を実行
				/*
				 * β= r'r'/rLDLr
				 * p = r' + βp
				 */
				double beta = rrNew / rr;
				this.p.SetAdded(this.r, this.p, beta);
			}
		}
	}
}