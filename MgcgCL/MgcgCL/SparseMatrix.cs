namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 疎行列
	/// </summary>
	public class SparseMatrix
	{
		/// <summary>
		/// 各要素
		/// </summary>
		readonly double[,] elements;

		/// <summary>
		/// その要素の列番号
		/// </summary>
		readonly long[,] columnIndeces;

		/// <summary>
		/// その行の非ゼロ要素数
		/// </summary>
		readonly long[] nonzeroCounts;

		/// <summary>
		/// 疎行列を生成する
		/// </summary>
		/// <param name="rowCount">行数</param>
		/// <param name="maxNonzeroCountPerRow">1行あたりの非ゼロ要素の最大数</param>
		public SparseMatrix(long rowCount, long maxNonzeroCountPerRow)
		{
			// 要素と列番号配列を初期化
			this.elements = new double[rowCount, maxNonzeroCountPerRow];
			this.columnIndeces = new long[rowCount, maxNonzeroCountPerRow];

			// 非ゼロ要素数を初期化
			this.nonzeroCounts = new long[rowCount];
		}

		/// <summary>
		/// ゼロ行列にする
		/// </summary>
		public void Clear()
		{
			// すべての非ゼロ要素数を
			for(long i = 0; i < nonzeroCounts.Length; i++)
			{
				// ゼロにする
				this.nonzeroCounts[i] = 0;
			}
		}

		/// <summary>
		/// 要素を取得または設定する
		/// </summary>
		/// <param name="i">行番号</param>
		/// <param name="j">列番号</param>
		/// <returns>i行目j列目の要素</returns>
		public double this[long i, long j]
		{
			// 取得
			get
			{
				// 要素番号を取得
				long k = this.GetElementIndex(i, j);

				// 有効な要素番号なら
				if(k >= 0)
				{
					// 要素の値を返す
					return this.elements[i, k];
				}

				// それ以外はゼロ
				return 0;
			}
			set
			{
				// 要素番号を取得
				long k = this.GetElementIndex(i, j);

				// 新しい要素なら
				if(k < 0)
				{
					// 設定する要素番号は最後尾にする
					k = this.nonzeroCounts[i];

					// 列番号を設定
					this.columnIndeces[i, k] = j;

					// 非ゼロ要素数を1つ増やす
					this.nonzeroCounts[i]++;
				}

				// その要素に値を設定
				this.elements[i, k] = value;

			}
		}

		/// <summary>
		/// 要素番号を返す
		/// </summary>
		/// <param name="i">行番号</param>
		/// <param name="j">列番号</param>
		/// <returns>要素配列にその要素の格納されている場所</returns>
		long GetElementIndex(long i, long j)
		{
			// その行のすべての非ゼロ要素に対して
			for(long k = 0; k < this.nonzeroCounts[i]; k++)
			{
				// 列番号が一致すれば
				if(this.columnIndeces[i, k] == j)
				{
					// その要素番号を返す
					return k;
				}
			}

			// それ以外は-1を返す
			return -1;
		}

		/// <summary>
		/// ベクトルとの乗法を計算し解ベクトルに設定する
		/// </summary>
		/// <param name="answer">演算結果を格納するベクトル</param>
		/// <param name="vector">右ベクトル</param>
		/// <param name="isEnabled">その要素が有効かどうかを表す配列</param>
		internal void Multiply(double[] answer, double[] vector, bool[] isEnabled)
		{
			// 各行について
			for(long i = 0; i < answer.Length; i++)
			{
				// 解をゼロに設定
				answer[i] = 0;

				// その行が有効なら
				if(isEnabled[i])
				{
					// その行のすべての非ゼロ要素に対して
					for(long k = 0; k < this.nonzeroCounts[i]; k++)
					{
						// 列番号を取得
						long j = this.columnIndeces[i, k];

						// その列が有効なら
						if(isEnabled[j])
						{
							// 解に積を加える
							answer[i] += this.elements[i, k] * vector[j];
						}
					}
				}
			}
		}
	}
}