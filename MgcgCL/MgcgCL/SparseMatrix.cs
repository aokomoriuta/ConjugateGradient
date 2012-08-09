using System.Threading.Tasks;
using System;
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
		internal readonly int[] ColumnIndeces;

		/// <summary>
		/// その行の非ゼロ要素数
		/// </summary>
		internal readonly int[] NonzeroCounts;

		/// <summary>
		/// 1行あたりの最大非ゼロ要素数
		/// </summary>
		public readonly int MaxNonzeroCountPerRow;

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
			this.NonzeroCounts = new int[rowCount];
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
			for(int i = 0; i < NonzeroCounts.Length; i++)
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
		double this[int i]
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
		public double this[int i, int j]
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
				int k = this.GetLocalIndex(i, j);

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
					int k = this.GetLocalIndex(i, j);

					// 新しい要素なら
					if(k < 0)
					{
						if(this.NonzeroCounts[i] == this.MaxNonzeroCountPerRow)
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
		int GetLocalIndex(int i, int j)
		{
			// 先頭を設定
			int first = i * this.MaxNonzeroCountPerRow;

			// その行のすべての非ゼロ要素に対して
			for(int k = 1; k < this.NonzeroCounts[i]; k++)
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
		int GetGlobalIndex(int i, int k)
		{
			return i * this.MaxNonzeroCountPerRow + k;
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
				for(int k = 0; k < this.NonzeroCounts[i]; k++)
				{
					// 配列番号を取得
					int index = this.GetGlobalIndex(i, k);

					// 列番号を取得
					int j = this.ColumnIndeces[index];

					// 解に積を加える
					answer[i] += this.Elements[index] * vector[j];
				}
			}
			//);
		}


		/// <summary>
		/// ヤコビ法で固有値を求める
		/// </summary>
		/// <param name="maxIteration">最大繰り返し回数</param>
		/// <param name="residual">収束値</param>
		/// <returns>固有値の配列</returns>
		/// <remarks>参考文献：物理のかぎしっぽ http://hooktail.org/computer/index.php?Jacobi%CB%A1 </remarks>
		public double[] GetEigenValues(int maxIteration, double residual)
		{
			// 対象の行列を生成
			double[,] A = new double[this.RowCount, this.RowCount];

			// 全行について
			for(int i = 0; i < this.RowCount; i++)
			{
				// 対角成分を設定
				A[i, i] = this.Elements[this.GetGlobalIndex(i, 0)];

				// その行のすべての非ゼロ要素に対して
				for(int k = 1; k < this.NonzeroCounts[i]; k++)
				{
					// 番号を取得
					int index = this.GetGlobalIndex(i, k);

					// 列番号を取得
					int j = this.ColumnIndeces[index];

					// 上側であれば
					if(i >= j)
					{
						// 行列の要素に設定
						A[i, j] = this.Elements[index];

						// 対称成分に設定
						A[j, i] = this.Elements[index];
					}
				}
			}


			// 収束したかどうか
			bool converged = false;

			// 最大値の行番号および列番号を初期化
			int p = 0;
			int q = 0;

			// 収束するまで繰り返す
			for(int iteration = 0; !converged; iteration++)
			{
				// 前の最大値の行番号および列番号を記憶
				int oldP = p;
				int oldQ = q;
				
				// 最大値を初期化
				double maxValue = residual / 10;

				// 全要素について
				for(int i = 0; i < this.RowCount; i++)
				{
					for(int j = 0; j < this.RowCount; j++)
					{
						// 非対角要素なら
						if(i != j)
						{
							// 要素の絶対値を取得
							double a_ij = Math.Abs(A[i, j]);

							// 最大値より大きければ
							if(a_ij > maxValue)
							{
								// その行番号と列番号を格納
								p = i;
								q = j;

								// 最大値に設定
								maxValue = a_ij;
							}
						}
					}
				}

				// 最大値が収束誤差以下だったり、これ以上変化しなくなったら
				converged = ((p == oldP) && (q == oldQ)) || (maxValue < residual);

				// 収束していなかったら
				if(!converged)
				{
					// 回転行列を生成
					//  * α = 1/2 (a_pp - a_qq)
					//  * β = -a_pq
					//  * γ = |α|/√(α^2 + β^2)
					//  * cosθ = √((1+γ)/2)
					//  * sinθ = √((1-γ)/2) sing(αβ)
					double alpha = (A[p, p] - A[q, q]) / 2;
					double beta = -A[p, q];
					double gamma = (double)(Math.Abs(alpha) / Math.Sqrt(alpha * alpha + beta * beta));
					double cos = (double)Math.Sqrt((1 + gamma) / 2);
					double sin = (double)Math.Sqrt((1 - gamma) / 2) * Math.Sign(alpha * beta);

					// 対象成分を取得
					double a_pp = A[p, p];
					double a_pq = A[p, q];
					double a_qq = A[q, q];

					// 全要素について
					Parallel.For(0, this.RowCount, (i) =>
					{
						// 他の行や列にも回転を実行
						//  * a_pi = a_pi cosθ - a_qi sinθ
						//  * a_qi = a_pi sinθ + a_qi cosθ
						double a_pi = A[p, i];
						double a_qi = A[q, i];
						A[p, i] = a_pi * cos - a_qi * sin;
						A[q, i] = a_pi * sin + a_qi * cos;

						// 対称成分も同様
						//  * a_ip = a_ip cosθ - a_iq sinθ
						//  * a_iq = a_ip sinθ + a_iq cosθ
						double a_ip = A[i, p];
						double a_iq = A[i, q];
						A[i, p] = a_ip * cos - a_iq * sin;
						A[i, q] = a_ip * sin + a_iq * cos;
					});

					// 対象の非対角成分をゼロにする
					A[p, p] = cos * (a_pp * cos - a_pq * sin) - sin * (a_pq * cos - a_qq * sin);
					A[p, q] = sin * (a_pp * cos - a_pq * sin) + cos * (a_pq * cos - a_qq * sin);
					A[q, p] = A[p, q];
					A[q, q] = sin * (a_pp * sin + a_pq * cos) + cos * (a_pq * sin + a_qq * cos);
				}
			}

			// 固有値配列を生成
			var eigenValues = new double[this.RowCount];

			// 全行の
			for(int i = 0; i < this.RowCount; i++)
			{
				// 対角成分が固有値
				eigenValues[i] = A[i, i];
			}

			// 固有値を返す
			return eigenValues;
		}

		/// <summary>
		/// 行数を取得する
		/// </summary>
		public int RowCount
		{
			get
			{
				// 非ゼロ要素数の灰列数
				return this.NonzeroCounts.Length;
			}
		}
	}
}