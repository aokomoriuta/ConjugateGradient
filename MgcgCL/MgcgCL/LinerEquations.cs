namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 連立線形方程式
	/// </summary>
	public class LinerEquations
	{
		/// <summary>
		/// 係数行列
		/// </summary>
		public readonly SparseMatrix A;

		/// <summary>
		/// 未知数ベクトル
		/// </summary>
		public double[] x;

		/// <summary>
		/// 右辺ベクトル
		/// </summary>
		public readonly double[] b;


		/// <summary>
		/// 連立線形方程式を作成する
		/// </summary>
		/// <param name="count">未知数の数</param>
		/// <param name="maxNonZeroCount">0でない要素の最大数</param>
		public LinerEquations(int count, int maxNonZeroCount)
		{
			// 係数行列・未知数・右辺ベクトルを初期化
			this.A = new SparseMatrix(count, maxNonZeroCount);
			this.x = new double[count];
			this.b = new double[count];
		}

		/// <summary>
		/// 未知数の数を取得する
		/// </summary>
		public int Count
		{
			get
			{
				return this.x.Length;
			}
		}
	}
}