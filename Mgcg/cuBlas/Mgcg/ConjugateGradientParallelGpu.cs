using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace LWisteria.Mgcg
{
	/// <summary>
	/// 複数GPUで共役勾配法
	/// </summary>
	public class ConjugateGradientParallelGpu : ConjugateGradientGpu
	{
		/// <summary>
		/// 使用するデバイスの番号
		/// </summary>
		const int DEVICE_ID = 1;

		readonly int deviceCount;

		/// <summary>
		/// cublasのハンドル
		/// </summary>
		IntPtr[] cublas;

		/// <summary>
		/// cusparseのハンドル
		/// </summary>
		IntPtr[] cusparse;

		/// <summary>
		/// matDescrのハンドル
		/// </summary>
		IntPtr[] matDescr;

		#region バッファー
		/// <summary>
		/// 係数行列
		/// </summary>
		VectorDouble[] vectorA;

		/// <summary>
		/// 列番号
		/// </summary>
		VectorInt[] vectorColumnIndeces;

		/// <summary>
		/// 行の先頭位置
		/// </summary>
		VectorInt[] vectorRowOffsets;

		/// <summary>
		/// 右辺ベクトル
		/// </summary>
		VectorDouble[] vectorB;

		/// <summary>
		/// 未知数
		/// </summary>
		VectorDouble[] vectorX;

		/// <summary>
		/// 係数行列と探索方向ベクトルの積
		/// </summary>
		VectorDouble[] vectorAp;

		/// <summary>
		/// 探索方向
		/// </summary>
		VectorDouble[] vectorP;

		/// <summary>
		/// 残差
		/// </summary>
		VectorDouble[] vectorR;
		#endregion

		#region バインディング
		/// <summary>
		/// デバイス数を取得する
		/// </summary>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "GetDeviceCount")]
		static extern int GetDeviceCount();

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
		/// 行列とベクトルの積
		/// </summary>
		/// <param name="cusparse"></param>
		/// <param name="matDescr"></param>
		/// <param name="deviceID"></param>
		/// <param name="y"></param>
		/// <param name="elements"></param>
		/// <param name="rowOffsets"></param>
		/// <param name="columnIndeces"></param>
		/// <param name="x"></param>
		/// <param name="elementsCount"></param>
		/// <param name="count"></param>
		/// <param name="alpha"></param>
		/// <param name="beta"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CsrMV")]
		static extern void CsrMV(IntPtr cusparse, IntPtr matDescr, int deviceID,
			IntPtr y,
			IntPtr elements, IntPtr rowOffsets, IntPtr columnIndeces,
			IntPtr x,
			int elementsCount, int count,
			double alpha, double beta);

		/// <summary>
		/// ベクトル同士の和
		/// </summary>
		/// <param name="cublas"></param>
		/// <param name="deviceID"></param>
		/// <param name="y"></param>
		/// <param name="x"></param>
		/// <param name="count"></param>
		/// <param name="alpha"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Axpy")]
		static extern void Axpy(IntPtr cublas, int deviceID,
			IntPtr y, IntPtr x,
			int count, double alpha);

		/// <summary>
		/// ベクトルの内積
		/// </summary>
		/// <param name="cublas"></param>
		/// <param name="deviceID"></param>
		/// <param name="y"></param>
		/// <param name="x"></param>
		/// <param name="count"></param>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Dot")]
		static extern double Dot(IntPtr cublas, int deviceID,
			IntPtr y, IntPtr x,
			int count);

		/// <summary>
		/// ベクトルのスカラー倍
		/// </summary>
		/// <param name="cublas"></param>
		/// <param name="deviceID"></param>
		/// <param name="x"></param>
		/// <param name="alpha"></param>
		/// <param name="count"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Scal")]
		static extern void Scal(IntPtr cublas, int deviceID,
			IntPtr x, double alpha,
			int count);

		/// <summary>
		/// ベクトルを複製する
		/// </summary>
		/// <param name="cublas"></param>
		/// <param name="deviceID"></param>
		/// <param name="y"></param>
		/// <param name="x"></param>
		/// <param name="count"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Copy")]
		static extern void Copy(IntPtr cublas, int deviceID,
			IntPtr y, IntPtr x,
			int count);
		#endregion

		/// <summary>
		/// 複数GPUでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientParallelGpu(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{
			// デバイス数を取得
			deviceCount = GetDeviceCount();

			// 配列を初期化
			cublas = new IntPtr[deviceCount];
			cusparse = new IntPtr[deviceCount];
			matDescr = new IntPtr[deviceCount];
			vectorA = new VectorDouble[deviceCount];
			vectorColumnIndeces = new VectorInt[deviceCount];
			vectorRowOffsets = new VectorInt[deviceCount];
			vectorX = new VectorDouble[deviceCount];
			vectorB = new VectorDouble[deviceCount];
			vectorAp = new VectorDouble[deviceCount];
			vectorP = new VectorDouble[deviceCount];
			vectorR = new VectorDouble[deviceCount];

			// 全デバイスで
			Parallel.For(0, deviceCount, deviceID =>
			{
				// cudaの使用準備
				cublas[deviceID] = CreateBlas(deviceID);
				cusparse[deviceID] = CreateSparse(deviceID);
				matDescr[deviceID] = CreateMatDescr(deviceID);

				// 行列を初期化
				vectorA[deviceID] = new VectorDouble(count * maxNonZeroCount, deviceID);
				vectorColumnIndeces[deviceID] = new VectorInt(count * maxNonZeroCount, deviceID);
				vectorRowOffsets[deviceID] = new VectorInt(count + 1, deviceID);

				// ベクトルを初期化
				vectorX[deviceID] = new VectorDouble(count, deviceID);
				vectorB[deviceID] = new VectorDouble(count, deviceID);
				vectorAp[deviceID] = new VectorDouble(count, deviceID);
				vectorP[deviceID] = new VectorDouble(count, deviceID);
				vectorR[deviceID] = new VectorDouble(count, deviceID);
			});
		}

		/// <summary>
		/// 複数GPUでの共役勾配法を廃棄する
		/// </summary>
		~ConjugateGradientParallelGpu()
		{
			// 全デバイスで
			Parallel.For(0, deviceCount, deviceID =>
			{
				// cublasとcusparseを廃棄
				DestroyBlas(cublas[deviceID], deviceID);
				DestroySparse(cusparse[deviceID], deviceID);
				DestroyMatDescr(matDescr[deviceID], deviceID);
			});
		}

		/// <summary>
		/// 初期化処理（データの転送など）
		/// </summary>
		public override void Initialize()
		{
			// 非ゼロ要素数を取得
			var nonzeroCount = A.RowOffsets[Count];
			Parallel.For(0, deviceCount, deviceID =>
			{
				// 行列を転送
				vectorA[deviceID].CopyFrom(A.Elements, nonzeroCount);
				vectorColumnIndeces[deviceID].CopyFrom(A.ColumnIndeces, nonzeroCount);
				vectorRowOffsets[deviceID].CopyFrom(A.RowOffsets, Count + 1);

				// ベクトルを転送
				vectorB[deviceID].CopyFrom(b, Count);
				vectorX[deviceID].CopyFrom(x, Count);
			});
		}

		/// <summary>
		/// 方程式を解く
		/// </summary>
		public override void Solve()
		{
			// 非ゼロ要素数を取得
			var elementsCount = A.RowOffsets[Count];

			// 生ポインタに変換
			var elements = vectorA[DEVICE_ID].ToRawPtr();
			var columnIndeces = vectorColumnIndeces[DEVICE_ID].ToRawPtr();
			var rowOffsets = vectorRowOffsets[DEVICE_ID].ToRawPtr();
			var x = vectorX[DEVICE_ID].ToRawPtr();
			var b = vectorB[DEVICE_ID].ToRawPtr();
			var Ap = vectorAp[DEVICE_ID].ToRawPtr();
			var p = vectorP[DEVICE_ID].ToRawPtr();
			var r = vectorR[DEVICE_ID].ToRawPtr();

			var deviceID = DEVICE_ID;
			var count = Count;

			// 方程式を解く
			{
				// 初期値を設定
				/*
				* (Ap)_0 = A * x
				* r_0 = b - Ap
				* rr_0 = r_0・r_0
				* p_0 = r_0
				*/
				CsrMV(cusparse[DEVICE_ID], matDescr[DEVICE_ID], deviceID, Ap, elements, rowOffsets, columnIndeces, x, elementsCount, count, 1, 0);
				Copy(cublas[DEVICE_ID], deviceID, r, b, count);
				Axpy(cublas[DEVICE_ID], deviceID, r, Ap, count, -1);
				Copy(cublas[DEVICE_ID], deviceID, p, r, count);
				double rr = Dot(cublas[DEVICE_ID], deviceID, r, r, count);
				
				// 収束しない間繰り返す
				for(Iteration = 0; ; Iteration++)
				{
					// 計算を実行
					/*
					* Ap = A * p
					* α = rr/(p・Ap)
					* x' += αp
					* r' -= αAp
					* r'r' = r'・r'
					*/
					CsrMV(cusparse[DEVICE_ID], matDescr[DEVICE_ID], deviceID, Ap, elements, rowOffsets, columnIndeces, p, elementsCount, count, 1, 0);
					double alpha = rr / Dot(cublas[DEVICE_ID], deviceID, p, Ap, count);
					Axpy(cublas[DEVICE_ID], deviceID, x, p, count, alpha);
					Axpy(cublas[DEVICE_ID], deviceID, r, Ap, count, -alpha);
					double rrNew = Dot(cublas[DEVICE_ID], deviceID, r, r, count);

					// 誤差を設定
					Residual = System.Math.Sqrt(rrNew);

					// 収束したら
					if(IsConverged)
					{
						// 繰り返し終了
						break;
					}
					else
					{
						// 残りの計算を実行
						/*
						* β= r'r'/rLDLr
						* p = r' + βp
						*/
						double beta = rrNew / rr;
						Scal(cublas[DEVICE_ID], deviceID, p, beta, count); Axpy(cublas[DEVICE_ID], deviceID, p, r, count, 1);

						rr = rrNew;
					}
				}
			}
		}

		/// <summary>
		/// 結果を読み込む
		/// </summary>
		public override void Read()
		{
			vectorX[DEVICE_ID].CopyTo(ref x, Count);
		}
	}
}