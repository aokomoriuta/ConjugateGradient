using System.Collections.Generic;

namespace LWisteria.StudiesOfOpenCL.SimpleViennaCL
{
	/// <summary>
	/// CSR形式の疎行列
	/// </summary>
	class CompuressedMatrix
	{
		/// <summary>
		/// 行列の要素
		/// </summary>
		public List<Dictionary<uint, double>> Elements { get; private set; }

		/// <summary>
		/// 疎行列を生成する
		/// </summary>
		public CompuressedMatrix()
		{
			Elements = new List<Dictionary<uint, double>>();
		}

		/// <summary>
		/// 要素を取得または設定する
		/// </summary>
		/// <param name="i"></param>
		/// <param name="j"></param>
		/// <returns></returns>
		public double this[int i, int j]
		{
			get
			{
				// 行数以上なら
				if(i > Elements.Count)
				{
					// ゼロ
					return 0;
				}

				// 要素を取得して返す
				double value = 0;
				Elements[i].TryGetValue((uint)j, out value);
				return value;
			}

			set
			{
				// 指定された行が設定されるまで
				while(i >= Elements.Count)
				{
					// 行を増やす
					Elements.Add(new Dictionary<uint, double>());
				}

				// 要素がすでにあったら
				if(Elements[i].ContainsKey((uint)j))
				{
					// その要素を設定
					Elements[i][(uint)j] = value;
				}
				// なかったら
				else
				{
					// 新しく追加
					Elements[i].Add((uint)j, value);
				}
			}
		}
	}
}