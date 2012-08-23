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
		public float[] x;

		/// <summary>
		/// 生成項
		/// </summary>
		public readonly float[] b;


		/// <summary>
		/// 連立線形方程式を作成する
		/// </summary>
		/// <param name="count">未知数の数</param>
		/// <param name="maxNonZeroCount">0でない要素の最大数</param>
		public LinerEquations(int count, int maxNonZeroCount)
		{
			// 係数行列・未知数・生成項を初期化
			this.A = new SparseMatrix(count, maxNonZeroCount);
			this.x = new float[count];
			this.b = new float[count];
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