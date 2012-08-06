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
		internal readonly double[] Elements;

		/// <summary>
		/// その要素の列番号
		/// </summary>
		internal readonly long[] ColumnIndeces;

		/// <summary>
		/// その行の非ゼロ要素数
		/// </summary>
		internal readonly long[] NonzeroCounts;

		/// <summary>
		/// 1行あたりの最大非ゼロ要素数
		/// </summary>
		public readonly long MaxNonzeroCountPerRow;

		/// <summary>
		/// 疎行列を生成する
		/// </summary>
		/// <param name="rowCount">行数</param>
		/// <param name="maxNonzeroCountPerRow">1行あたりの非ゼロ要素の最大数</param>
		public SparseMatrix(long rowCount, long maxNonzeroCountPerRow)
		{
			// 要素と列番号配列を初期化
			this.Elements = new double[rowCount* maxNonzeroCountPerRow];
			this.ColumnIndeces = new long[rowCount* maxNonzeroCountPerRow];

			// 非ゼロ要素数を初期化
			this.NonzeroCounts = new long[rowCount];
			this.Clear();
			
			// 1行あたりの最大非ゼロ要素数を設定
			this.MaxNonzeroCountPerRow = maxNonzeroCountPerRow;
		}

		/// <summary>
		/// ゼロ行列にする
		/// </summary>
		public void Clear()
		{
			// すべての非ゼロ要素数を
			for(long i = 0; i < NonzeroCounts.Length; i++)
			{
				// 1にする
				this.NonzeroCounts[i] = 1;

				// 対角成分を0にする
				this[i] = 0;
			}
		}

		/// <summary>
		/// 対角成分を取得または設定する
		/// </summary>
		/// <param name="i">行番号および列番号</param>
		/// <returns>対角成分</returns>
		double this[long i]
		{
			// 取得
			get
			{
				// 先頭要素を返す
				return this.Elements[i * this.MaxNonzeroCountPerRow];
			}
			// 設定
			set
			{
				// 先頭要素に設定
				this.Elements[i * this.MaxNonzeroCountPerRow] = value;

				// 列番号を設定
				this.ColumnIndeces[i * this.MaxNonzeroCountPerRow] = i;
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
				// 対角成分なら
				if(i == j)
				{
					// 対角成分を返す
					return this[i];
				}

				// 要素番号を取得
				long k = this.GetLocalIndex(i, j);

				// 有効な要素番号なら
				if(k >= 0)
				{
					// 要素の値を返す
					return this.Elements[this.GetGlobalIndex(i, k)];
				}

				// それ以外はゼロ
				return 0;
			}
			set
			{
				// 対角成分なら
				if(i == j)
				{
					// 対角成分を設定
					this[i] = value;
				}
				// 違う場合は
				else
				{
					// 要素番号を取得
					long k = 0;// this.GetLocalIndex(i, j);

					// 新しい要素なら
					//if(k < 0)
					{
						if(this.NonzeroCounts[i] == this.MaxNonzeroCountPerRow - 1)
						{
							throw new System.IndexOutOfRangeException(string.Format("1行に格納できる最大列数を超えました＠{0}行目", i));
						}

						// 設定する要素番号は最後尾にする
						k = this.NonzeroCounts[i];

						// 列番号を設定
						this.ColumnIndeces[this.GetGlobalIndex(i, k)] = j;

						// 非ゼロ要素数を1つ増やす
						this.NonzeroCounts[i]++;
					}

					// その要素に値を設定
					this.Elements[this.GetGlobalIndex(i, k)] = value;
				}
			}
		}

		/// <summary>
		/// その行での要素番号を返す
		/// </summary>
		/// <param name="i">行番号</param>
		/// <param name="j">列番号</param>
		/// <returns>要素配列にその要素の格納されている場所</returns>
		long GetLocalIndex(long i, long j)
		{

			System.Console.WriteLine("{0}, {1}", i, j);

			// 先頭を設定
			long first = i * this.MaxNonzeroCountPerRow;

			// その行のすべての非ゼロ要素に対して
			for(long k = 1; k < this.NonzeroCounts[i]; k++)
			{
				// 列番号が一致すれば
				if(this.ColumnIndeces[first + k] == j)
				{
					// その要素番号を返す
					return k;
				}
			}

			// それ以外は-1を返す
			return -1;
		}

		/// <summary>
		/// 全体での要素番号を返す
		/// </summary>
		/// <param name="i">行番号</param>
		/// <param name="k">行内での要素番号</param>
		/// <returns>全体での要素番号</returns>
		long GetGlobalIndex(long i, long k)
		{
			return i * this.MaxNonzeroCountPerRow + k;
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
					for(long k = 0; k < this.NonzeroCounts[i]; k++)
					{
						// 列番号を取得
						long j = this.ColumnIndeces[this.GetGlobalIndex(i, k)];

						// その列が有効なら
						if(isEnabled[j])
						{
							// 解に積を加える
							answer[i] += this.Elements[this.GetGlobalIndex(i, k)] * vector[j];
						}
					}
				}
			}
		}
	}
}