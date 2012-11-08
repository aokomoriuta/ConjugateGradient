using System.Threading.Tasks;
using System;
namespace LWisteria.Mgcg
{
	/// <summary>
	/// 疎行列
	/// </summary>
	public class SparseMatrix
	{
		/// <summary>
		/// 各要素
		/// </summary>
		internal readonly double[] Elements;

		/// <summary>
		/// その要素の列番号
		/// </summary>
		internal readonly int[] ColumnIndeces;

		/// <summary>
		/// その行の先頭位置
		/// </summary>
		internal readonly int[] RowOffsets;

		/// <summary>
		/// 疎行列を生成する
		/// </summary>
		/// <param name="rowCount">行数</param>
		/// <param name="maxNonzeroCountPerRow">1行あたりの非ゼロ要素の最大数</param>
		public SparseMatrix(int rowCount, int maxNonzeroCountPerRow)
		{
			// 要素と列番号配列を初期化
			this.Elements = new double[rowCount* maxNonzeroCountPerRow];
			this.ColumnIndeces = new int[rowCount* maxNonzeroCountPerRow];

			// 非ゼロ要素数を初期化
			this.RowOffsets = new int[rowCount + 1];
			this.Clear();
		}

		/// <summary>
		/// ゼロ行列にする
		/// </summary>
		public void Clear()
		{
			// すべての行の先頭位置を
			for(int i = 0; i < RowOffsets.Length; i++)
			{
				// 0にする
				this.RowOffsets[i] = 0;
			}

			// すべての要素を
			for(int i = 0; i < Elements.Length; i++)
			{
				// 初期化する
				Elements[i] = 0;
				ColumnIndeces[i] = -1;
			}
		}

		/// <summary>
		/// ベクトルとの乗法を計算し解ベクトルに設定する
		/// </summary>
		/// <param name="answer">演算結果を格納するベクトル</param>
		/// <param name="vector">右ベクトル</param>
		/// <param name="isEnabled">その要素が有効かどうかを表す配列</param>
		internal void Multiply(double[] answer, double[] vector)
		{
			// 各行について
			//Parallel.For(0, answer.Length, (i) =>
			for(int i = 0; i < answer.Length; i++)
			{
				// 解をゼロに設定
				answer[i] = 0;

				// その行のすべての非ゼロ要素に対して
				for(int k = this.RowOffsets[i]; k < RowOffsets[i + 1] ; k++)
				{
					// 列番号を取得
					int j = this.ColumnIndeces[k];

					// 解に積を加える
					answer[i] += this.Elements[k] * vector[j];
				}
			}
			//);
		}

		/// <summary>
		/// 行数を取得する
		/// </summary>
		public int RowCount
		{
			get
			{
				// 非ゼロ要素数の灰列数
				return this.RowOffsets.Length;
			}
		}
	}
}