using System.Threading.Tasks;
namespace LWisteria.MgcgCL.LongVector
{
	/// <summary>
	/// 多次元ベクトルに対する拡張メソッド群
	/// </summary>
	static class LongVectorExtensions
	{
		/// <summary>
		/// ベクトルの内積を計算する
		/// </summary>
		/// <param name="left">左ベクトル</param>
		/// <param name="right">右ベクトル</param>
		/// <returns>各要素の積和</returns>
		public static float Dot(this float[] left, float[] right)
		{
			// 解
			float answer = 0;

			// 全要素について
			//Parallel.For(0, left.Length, (i)=>
			for(int i = 0; i < left.Length; i++)
			{
				// 各要素をかけたものを解に追加
				answer += left[i] * right[i];
			}
			//);

			// 解を返す
			return answer;
		}


		/// <summary>
		/// 右ベクトルにスカラー係数をかけてのベクトルの加算を計算し、結果を解に格納する
		/// </summary>
		/// <param name="answer">演算結果を格納するベクトル</param>
		/// <param name="left">左ベクトル</param>
		/// <param name="right">右ベクトル</param>
		/// <param name="a">係数</param>
		public static void SetAdded(this float[] answer, float[] left, float[] right, float a)
		{
			// 全要素について
			//Parallel.For(0, left.Length, (i) =>
			for(int i = 0; i < answer.Length; i++)
			{
				// 答えベクトルを計算
				answer[i] = left[i] + a * right[i];
			}
			//);
		}

		/// <summary>
		/// 要素の中で最大値を探す
		/// </summary>
		/// <param name="vector">探索するベクトル</param>
		/// <returns>最大値</returns>
		public static float Max(this float[] vector)
		{
			// 最初の要素で最大値を初期化
			float max = System.Math.Abs(vector[0]);

			// 残りの要素のうち
			for(int i = 1; i < vector.Length; i++)
			{
				// 最大のものを探す
				max = System.Math.Max(System.Math.Abs(vector[i]), max);
			}

			// 最大値を返す
			return max;
		}
	}
}