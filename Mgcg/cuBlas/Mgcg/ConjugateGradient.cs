namespace LWisteria.Mgcg
{
	/// <summary>
	/// 共役勾配法
	/// </summary>
	abstract public class ConjugateGradient : LinerEquations
	{
		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		public readonly int MinIteration;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		public readonly int MaxIteration;

		/// <summary>
		/// 収束判定誤差
		/// </summary>
		public readonly double AllowableResidual;

		/// <summary>
		/// 繰り返し回数
		/// </summary>
		public int Iteration { get; protected set; }

		/// <summary>
		/// 現在の誤差
		/// </summary>
		public double Residual { get; protected set; }

		/// <summary>
		/// 共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount">最大非ゼロ要素数</param>
		/// <param name="_minIteration">最小繰り返し回数</param>
		/// <param name="_maxIteration">最大繰り返し回数</param>
		/// <param name="_allowableResidual">収束誤差</param>
		public ConjugateGradient(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double _allowableResidual)
			: base(count, maxNonZeroCount)
		{
			// 最小・最大繰り返し回数を設定
			this.MinIteration = _minIteration;
			this.MaxIteration = _maxIteration;

			// 収束判定誤差を設定
			this.AllowableResidual = _allowableResidual;
		}

		/// <summary>
		/// 収束しているかどうか
		/// </summary>
		/// <returns>繰り返し回数と収束誤差から判定した結果</returns>
		protected bool IsConverged
		{
			get
			{
				// 誤差表示
				System.Console.WriteLine("{2} {0,3}: {1:E}", this.Iteration, this.Residual, this.GetType().Name);

				// 最小繰り返し回数未満なら
				if(this.Iteration < this.MinIteration)
				{
					// 収束していない
					return false;
				}
				// 最大繰り返し回数を超えていたら
				else if(this.Iteration > this.MaxIteration)
				{
					// 例外
					throw new System.ApplicationException("圧力方程式が収束しませんでした。");
				}

				// 誤差が収束判定誤差より小さいかどうかを計算
				return (this.Residual < this.AllowableResidual);
			}
		}
		/// <summary>
		/// 方程式を解く
		/// </summary>
		public abstract void Solve();
	}
}