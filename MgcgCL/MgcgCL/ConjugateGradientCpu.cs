using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// CPUでの共役勾配法
	/// </summary>
	public class ConjugateGradientCpu : ConjugateGradient
	{
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
		/// CPUでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientCpu(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
	{
			// 残差および探索方向と係数行列と探索方向の積を初期化
			this.r = new double[count];
			this.p = new double[count];
			this.Ap = new double[count];
		}

		/// <summary>
		/// CPUで方程式を解く
		/// </summary>
		override public void Solve()
		{
			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = (LDLr)_0
			 */
			this.A.Multiply(this.Ap, this.x);
			this.r.SetAdded(this.b, this.x, -1);
			this.r.CopyTo(this.p, 0);

			// 収束したかどうか
			bool converged = false;

			// 収束しない間繰り返す
			for(this.Iteration = 0; !converged; this.Iteration++)
			{
				// 計算を実行
				/*
				 * rr = r・r
				 * Ap = A * p
				 * α = rr/(p・Ap)
				 * x' += αp
				 * r' -= αAp
				 */
				double rr = this.r.Dot(this.r);
				this.A.Multiply(this.Ap, this.p);
				double alpha = rr / this.p.Dot(this.Ap);
				this.x.SetAdded(this.x, this.p, alpha);
				this.r.SetAdded(this.r, this.Ap, -alpha);

				// 収束したかどうかを取得
				converged = this.IsConverged(r.MaxAbsolute());

				// 収束していなかったら
				if(!converged)
				{
					// 残りの計算を実行
					/*
					 * r'r' = r'・r'
					 * β= r'r'/rLDLr
					 * p = r' + βp
					 */
					double rrNew = this.r.Dot(this.r);
					double beta = rrNew / rr;
					this.p.SetAdded(this.r, this.p, beta);
				}
			}
		}
	}
}