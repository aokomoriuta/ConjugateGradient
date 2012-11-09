using LWisteria.Mgcg.LongVector;

namespace LWisteria.Mgcg
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
			 * p_0 = r_0
			 * rr_0 = r_0・r_0
			 */
			this.A.Multiply(this.Ap, this.x);
			this.r.SetAdded(this.b, this.Ap, -1);
			this.r.CopyTo(this.p, 0);
			double rr = this.r.Dot(this.r);

			// 収束しない間繰り返す
			for(this.Iteration = 0; ; this.Iteration++)
			{
				// 計算を実行
				/*
				 * Ap = A * p
				 * α = rr/(p・Ap)
				 * x' += αp
				 * r' -= αAp
				 * r'r' = r'・r'
				 */
				this.A.Multiply(this.Ap, this.p);
				double alpha = rr / this.p.Dot(this.Ap);
				this.x.SetAdded(this.x, this.p, alpha);
				this.r.SetAdded(this.r, this.Ap, -alpha);
				double rrNew = this.r.Dot(this.r);

				// 誤差を設定
				this.Residual = System.Math.Sqrt(rrNew);

				// 収束していたら
				if(this.IsConverged)
				{
					// 繰り返し終了
					break;
				}
				// なかったら
				else
				{
					// 残りの計算を実行
					/*
					 * β= r'r'/rLDLr
					 * p = r' + βp
					 */
					double beta = rrNew / rr;
					this.p.SetAdded(this.r, this.p, beta);
					rr = rrNew;
				}
			}
		}
	}
}