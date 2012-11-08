using System;
using MathFunctions = System.Math;
using System.Runtime.InteropServices;
namespace LWisteria.Mgcg
{
	/// <summary>
	/// 1GPUで共役勾配法
	/// </summary>
	public class ConjugateGradientSingleGpu : ConjugateGradientGpu
	{
		/// <summary>
		/// 使用するデバイスの番号
		/// </summary>
		const int DEVICE_ID = 0;

		/// <summary>
		/// cublasのハンドル
		/// </summary>
		IntPtr cublas;

		/// <summary>
		/// cusparseのハンドル
		/// </summary>
		IntPtr cusparse;

		/// <summary>
		/// matDescrのハンドル
		/// </summary>
		IntPtr matDescr;

		#region バッファー
		/// <summary>
		/// 係数行列
		/// </summary>
		VectorDouble vectorA;

		/// <summary>
		/// 列番号
		/// </summary>
		VectorInt vectorColumnIndeces;

		/// <summary>
		/// 行の先頭位置
		/// </summary>
		VectorInt vectorRowOffsets;

		/// <summary>
		/// 右辺ベクトル
		/// </summary>
		VectorDouble vectorB;

		/// <summary>
		/// 未知数
		/// </summary>
		VectorDouble vectorX;

		/// <summary>
		/// 係数行列と探索方向ベクトルの積
		/// </summary>
		VectorDouble vectorAp;

		/// <summary>
		/// 探索方向
		/// </summary>
		VectorDouble vectorP;

		/// <summary>
		/// 残差
		/// </summary>
		VectorDouble vectorR;
		#endregion

		/// <summary>
		/// 方程式を解く
		/// </summary>
		/// <param name="cublas"></param>
		/// <param name="deviceID"></param>
		/// <param name="x"></param>
		/// <param name="b"></param>
		/// <param name="count"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Solve")]
		static extern void Solve(IntPtr cublas, IntPtr cusparse, IntPtr matDescr, int deviceID,
			IntPtr elementsVector, IntPtr rowOffsetsVector, IntPtr columnIndecesVector,
			IntPtr xVector, IntPtr bVector,
			IntPtr ApVector, IntPtr pVector, IntPtr rVector,
			int elementsCount, int count,
			double allowableResidual, int minIteration, int maxIteration,
			out int iteration, out double residual);

		/// <summary>
		/// 1GPUでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientSingleGpu(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{
			// cudaの使用準備
			cublas = CreateBlas(DEVICE_ID);
			cusparse = CreateSparse(DEVICE_ID);
			matDescr = CreateMatDescr(DEVICE_ID);

			// 行列を初期化
			vectorA = new VectorDouble(count * maxNonZeroCount, DEVICE_ID);
			vectorColumnIndeces = new VectorInt(count * maxNonZeroCount, DEVICE_ID);
			vectorRowOffsets = new VectorInt(count + 1, DEVICE_ID);

			// ベクトルを初期化
			vectorX = new VectorDouble(count, DEVICE_ID);
			vectorB = new VectorDouble(count, DEVICE_ID);
			vectorAp = new VectorDouble(count, DEVICE_ID);
			vectorP = new VectorDouble(count, DEVICE_ID);
			vectorR = new VectorDouble(count, DEVICE_ID);
		}

		/// <summary>
		/// 1GPUでの共役勾配法を廃棄する
		/// </summary>
		~ConjugateGradientSingleGpu()
		{
			// cublasとcusparseを廃棄
			DestroyBlas(cublas, DEVICE_ID);
			DestroySparse(cusparse, DEVICE_ID);
			DestroyMatDescr(matDescr, DEVICE_ID);
		}


		/// <summary>
		/// 初期化処理（データの転送など）
		/// </summary>
		public override void Initialize()
		{
			// 非ゼロ要素数を取得
			var nonzeroCount = A.RowOffsets[Count];

			// 行列を転送
			vectorA.CopyFrom(A.Elements, nonzeroCount);
			vectorColumnIndeces.CopyFrom(A.ColumnIndeces, nonzeroCount);
			vectorRowOffsets.CopyFrom(A.RowOffsets, Count + 1);

			// ベクトルを転送
			vectorB.CopyFrom(b, Count);
			vectorX.CopyFrom(x, Count);
		}

		/// <summary>
		/// 方程式を解く
		/// </summary>
		public override void Solve()
		{
			// 非ゼロ要素数を取得
			var nonzeroCount = A.RowOffsets[Count];
			
			// 方程式を解く
			int iteration;
			double residual;
			Solve(cublas, cusparse, matDescr, DEVICE_ID,
				vectorA.Ptr, vectorRowOffsets.Ptr, vectorColumnIndeces.Ptr,
				vectorX.Ptr, vectorB.Ptr,
				vectorAp.Ptr, vectorP.Ptr, vectorR.Ptr,
				nonzeroCount, Count,
				AllowableResidual, MinIteration, MaxIteration,
				out iteration, out residual);

			this.Iteration = iteration - 1;
			this.Residual = residual;
		}

		/// <summary>
		/// 結果を読み込む
		/// </summary>
		public override void Read()
		{
			vectorX.CopyTo(ref x, Count);
		}
	}
}