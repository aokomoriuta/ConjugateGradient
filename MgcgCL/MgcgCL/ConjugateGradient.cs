using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 共役勾配法
	/// </summary>
	abstract public class ConjugateGradient : LinerEquations
	{
		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		readonly int minIteration;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		readonly int maxIteration;

		/// <summary>
		/// 収束判定誤差
		/// </summary>
		readonly double allowableResidual;

		/// <summary>
		/// 繰り返し回数
		/// </summary>
		public int Iteration { get; protected set; }

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
			this.minIteration = _minIteration;
			this.maxIteration = _maxIteration;

			// 収束判定誤差を設定
			this.allowableResidual = _allowableResidual;
		}

		/// <summary>
		/// 収束しているかどうか
		/// </summary>
		/// <param name="r">誤差</param>
		/// <returns>繰り返し回数と収束誤差から判定した結果</returns>
		protected bool IsConverged(double r)
		{
			//Console.WriteLine("{2} {0,3}: {1:E}", this.Iteration, r, this.GetType().Name);

			// 最小繰り返し回数未満なら
			if(this.Iteration < this.minIteration)
			{
				// 収束していない
				return false;
			}
			// 最大繰り返し回数を超えていたら
			else if(this.Iteration > this.maxIteration)
			{
				// 例外
				throw new System.ApplicationException("圧力方程式が収束しませんでした。");
			}

			// 誤差が収束判定誤差より小さいかどうかを計算
			return (r < this.allowableResidual);
		}
		/// <summary>
		/// 方程式を解く
		/// </summary>
		public abstract void Solve();
	}
}