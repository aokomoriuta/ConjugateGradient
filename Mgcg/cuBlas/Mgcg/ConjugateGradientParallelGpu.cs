using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Linq;

namespace LWisteria.Mgcg
{
	/// <summary>
	/// 複数GPUで共役勾配法
	/// </summary>
	public class ConjugateGradientParallelGpu : ConjugateGradientGpu
	{
		/// <summary>
		/// デバイス数
		/// </summary>
		readonly int deviceCount;

		/// <summary>
		/// デバイスが処理する要素の先頭位置
		/// </summary>
		readonly int[] offsetsForDevice;

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
		VectorDouble[] vectorElements;

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

		/// <summary>
		/// ホスト側の一時変数
		/// </summary>
		double[] bufferHost;

		/// <summary>
		/// 各デバイスでの内積の結果
		/// </summary>
		double[] resultsDot;

		/// <summary>
		/// 各デバイスが計算する範囲の列の最小値
		/// </summary>
		int[] minJ;

		/// <summary>
		/// 各デバイスが計算する範囲の列の最大値
		/// </summary>
		int[] maxJ;
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
			int elementsCount, int rowCount, int columnCount,
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
			int count, int yOffset, int xOffset);

		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Initialize")]
		static extern void Initialize(
			double[] elements, int[] rowOffsets, int[] columnIndeces,
			double[] x, double[] b,
			IntPtr elementsVector, IntPtr rowOffsetsVector, IntPtr columnIndecesVector,
			IntPtr xVector, IntPtr bVector,
			IntPtr pVector,
			out int minJ, out int maxJ,
			int count,
			int countForDevice, int offsetForDevice, int elementCountForDevice, int elementOffsetForDevice);


		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "P2Host")]
		static extern void P2Host(IntPtr pVector, double[] p,
			int thisCount, int thisOffset,
			int lastCount, int nextCount);


		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "P2Device")]
		static extern void P2Device(IntPtr pVector, double[] p,
			int thisCount, int thisOffset,
			int lastCount, int nextCount);


		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Solve0")]
		static extern double Solve0(IntPtr cublas, IntPtr cusparse, IntPtr matDescr,
			IntPtr elementsVector, IntPtr rowOffsetsVector, IntPtr columnIndecesVector,
			IntPtr xVector, IntPtr bVector,
			IntPtr ApVector, IntPtr pVector, IntPtr rVector,
			int count,
			int countForDevice, int offsetForDevice, int elementCountForDevice);


		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Solve1")]
		static extern double Solve1(IntPtr cublas, IntPtr cusparse, IntPtr matDescr,
			IntPtr elementsVector, IntPtr rowOffsetsVector, IntPtr columnIndecesVector,
			IntPtr ApVector, IntPtr pVector,
			int count,
			int countForDevice, int offsetForDevice, int elementCountForDevice);


		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Solve2")]
		static extern double Solve2(IntPtr cublas, double alpha,
			IntPtr xVector,
			IntPtr ApVector, IntPtr pVector, IntPtr rVector,
			int countForDevice, int offsetForDevice);


		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Solve3")]
		static extern void Solve3(IntPtr cublas, double beta,
			IntPtr pVector, IntPtr rVector,
			int countForDevice, int offsetForDevice);
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

			// デバイスが計算する要素の先頭位置を生成
			offsetsForDevice = new int[deviceCount + 1];
			offsetsForDevice[0] = 0;
			for(int i = 1; i < deviceCount; i++)
			{
				offsetsForDevice[i] = offsetsForDevice[i - 1] + (int)Math.Floor((double)Count / deviceCount);
			}
			offsetsForDevice[deviceCount] = Count;

			// 内積の結果を初期化
			resultsDot = new double[deviceCount];

			bufferHost = new double[Count];
			minJ = new int[deviceCount];
			maxJ = new int[deviceCount];

			// 配列を初期化
			cublas = new IntPtr[deviceCount];
			cusparse = new IntPtr[deviceCount];
			matDescr = new IntPtr[deviceCount];
			vectorElements = new VectorDouble[deviceCount];
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
				SetDevice(deviceID);

				var countForDevice = CountForDevice(deviceID);

				// cudaの使用準備
				cublas[deviceID] = CreateBlas();
				cusparse[deviceID] = CreateSparse();
				matDescr[deviceID] = CreateMatDescr();

				// 行列を初期化
				vectorElements[deviceID] = new VectorDouble(countForDevice * maxNonZeroCount);
				vectorColumnIndeces[deviceID] = new VectorInt(countForDevice * maxNonZeroCount);
				vectorRowOffsets[deviceID] = new VectorInt(countForDevice + 1);

				// ベクトルを初期化
				vectorX[deviceID] = new VectorDouble(countForDevice);
				vectorB[deviceID] = new VectorDouble(countForDevice);
				vectorAp[deviceID] = new VectorDouble(countForDevice);
				vectorP[deviceID] = new VectorDouble(count);
				vectorR[deviceID] = new VectorDouble(countForDevice);
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
				SetDevice(deviceID);

				// 行列を廃棄
				vectorElements[deviceID].Dispose();
				vectorColumnIndeces[deviceID].Dispose();
				vectorRowOffsets[deviceID].Dispose();

				// ベクトルを廃棄
				vectorX[deviceID].Dispose();
				vectorB[deviceID].Dispose();
				vectorAp[deviceID].Dispose();
				vectorP[deviceID].Dispose();
				vectorR[deviceID].Dispose();

				// cublasとcusparseを廃棄
				DestroyBlas(cublas[deviceID]);
				DestroySparse(cusparse[deviceID]);
				DestroyMatDescr(matDescr[deviceID]);
			});
		}

		/// <summary>
		/// 初期化処理（データの転送など）
		/// </summary>
		public override void Initialize()
		{
			// 全デバイスで初期化
			Parallel.For(0, deviceCount, deviceID =>
			{
				SetDevice(deviceID);
				var countForDevice = CountForDevice(deviceID);
				var offsetForDevice = offsetsForDevice[deviceID];
				var elementCountForDevice = A.RowOffsets[offsetsForDevice[deviceID + 1]] - A.RowOffsets[offsetForDevice];
				var elementOffsetForDevice = A.RowOffsets[offsetForDevice];

				Initialize(
					A.Elements, A.RowOffsets, A.ColumnIndeces,
					x, b,
					vectorElements[deviceID].Ptr, vectorRowOffsets[deviceID].Ptr, vectorColumnIndeces[deviceID].Ptr,
					vectorX[deviceID].Ptr, vectorB[deviceID].Ptr,
					vectorP[deviceID].Ptr,
					out minJ[deviceID], out maxJ[deviceID],
					Count,
					countForDevice, offsetForDevice, elementCountForDevice, elementOffsetForDevice);
			});
		}

		/// <summary>
		/// 探索方向ベクトルを同期する
		/// </summary>
		void SyncP()
		{
			// 全デバイスで探索方向ベクトルの必要部分をホストに転送
			Parallel.For(0, deviceCount, deviceID =>
			{
				SetDevice(deviceID);

				var columnIndeces = vectorColumnIndeces[deviceID].Ptr;
				var b = vectorB[deviceID].Ptr;
				var p = vectorP[deviceID].Ptr;

				var countForDevice = CountForDevice(deviceID);
				var offsetForDevice = offsetsForDevice[deviceID];
				var lastCount = (deviceID > 0) ? offsetForDevice - minJ[deviceID] : 0;
				var nextCount = (deviceID < deviceCount - 1) ? maxJ[deviceID] - countForDevice - offsetForDevice + 1 : 0;

				P2Host(p, bufferHost, countForDevice, offsetForDevice, lastCount, nextCount);
			});

			// 全デバイスで探索方向ベクトルの必要部分をホストから転送
			Parallel.For(0, deviceCount, deviceID =>
			{
				SetDevice(deviceID);

				var columnIndeces = vectorColumnIndeces[deviceID].Ptr;
				var b = vectorB[deviceID].Ptr;
				var p = vectorP[deviceID].Ptr;

				var countForDevice = CountForDevice(deviceID);
				var offsetForDevice = offsetsForDevice[deviceID];
				var lastCount = (deviceID > 0) ? offsetForDevice - minJ[deviceID] : 0;
				var nextCount = (deviceID < deviceCount - 1) ? maxJ[deviceID] - countForDevice - offsetForDevice + 1 : 0;

				P2Device(p, bufferHost, countForDevice, offsetForDevice, lastCount, nextCount);
			});
		}

		/// <summary>
		/// 方程式を解く
		/// </summary>
		public override void Solve()
		{
			// 探索方向ベクトルを同期
			SyncP();

			// 全デバイスで初期値を設定
			/*
			* (Ap)_0 = A * x
			* r_0 = b - Ap
			* p_0 = r_0
			* rr_0 = r_0・r_0
			*/
			Parallel.For(0, deviceCount, deviceID =>
			{
				SetDevice(deviceID);

				var blas = cublas[deviceID];
				var sparse = cusparse[deviceID];
				var descr = matDescr[deviceID];
				var elements = vectorElements[deviceID].Ptr;
				var columnIndeces = vectorColumnIndeces[deviceID].Ptr;
				var rowOffsets = vectorRowOffsets[deviceID].Ptr;
				var x = vectorX[deviceID].Ptr;
				var b = vectorB[deviceID].Ptr;
				var Ap = vectorAp[deviceID].Ptr;
				var p = vectorP[deviceID].Ptr;
				var r = vectorR[deviceID].Ptr;

				var countForDevice = CountForDevice(deviceID);
				var offsetForDevice = offsetsForDevice[deviceID];
				var elementCountForDevice = A.RowOffsets[offsetsForDevice[deviceID + 1]] - A.RowOffsets[offsetsForDevice[deviceID]];

				resultsDot[deviceID] = Solve0(blas, sparse, descr,
					elements, rowOffsets, columnIndeces,
					x, b,
					Ap, p, r,
					Count,
					countForDevice, offsetForDevice, elementCountForDevice);
			});
			double rr = resultsDot.Sum();

			// 収束しない間繰り返す
			for(this.Iteration = 0; ; this.Iteration++)
			{
				// 探索方向ベクトルを同期
				SyncP();

				// 全デバイスでαを計算
				/*
				* Ap = A * p
				* α = rr/(p・Ap)
				*/
				Parallel.For(0, deviceCount, deviceID =>
				{
					SetDevice(deviceID);

					var blas = cublas[deviceID];
					var sparse = cusparse[deviceID];
					var descr = matDescr[deviceID];
					var elements = vectorElements[deviceID].Ptr;
					var columnIndeces = vectorColumnIndeces[deviceID].Ptr;
					var rowOffsets = vectorRowOffsets[deviceID].Ptr;
					var Ap = vectorAp[deviceID].Ptr;
					var p = vectorP[deviceID].Ptr;

					var countForDevice = CountForDevice(deviceID);
					var offsetForDevice = offsetsForDevice[deviceID];
					var elementCountForDevice = A.RowOffsets[offsetsForDevice[deviceID + 1]] - A.RowOffsets[offsetsForDevice[deviceID]];

					resultsDot[deviceID] = Solve1(blas, sparse, descr,
						elements, rowOffsets, columnIndeces,
						Ap, p,
						Count,
						countForDevice, offsetForDevice, elementCountForDevice);
				});
				double alpha = rr / resultsDot.Sum();

				// 全デバイスでこのステップの残差を計算
				/*
				 * x' += αp
				 * r' -= αAp
				 * r'r' = r'・r'
				 */
				Parallel.For(0, deviceCount, deviceID =>
				{
					SetDevice(deviceID);

					var blas = cublas[deviceID];
					var x = vectorX[deviceID].Ptr;
					var Ap = vectorAp[deviceID].Ptr;
					var p = vectorP[deviceID].Ptr;
					var r = vectorR[deviceID].Ptr;

					var countForDevice = CountForDevice(deviceID);
					var offsetForDevice = offsetsForDevice[deviceID];

					resultsDot[deviceID] = Solve2(blas, alpha,
						x,
						Ap, p, r,
						countForDevice, offsetForDevice);
				});
				double rrNew = resultsDot.Sum();

				// 誤差を設定
				this.Residual = System.Math.Sqrt(rrNew);

				// 収束していたら
				if(this.IsConverged)
				{
					// 繰り返し終了
					break;
				}
				// なかったら
				else
				{
					// 全デバイスで残りの計算を実行
					/*
					 * β= r'r'/rLDLr
					 * p = r' + βp
					 */
					double beta = rrNew / rr;
					Parallel.For(0, deviceCount, deviceID =>
					{
						SetDevice(deviceID);

						var blas = cublas[deviceID];
						var x = vectorX[deviceID].Ptr;
						var Ap = vectorAp[deviceID].Ptr;
						var p = vectorP[deviceID].Ptr;
						var r = vectorR[deviceID].Ptr;

						var countForDevice = CountForDevice(deviceID);
						var offsetForDevice = offsetsForDevice[deviceID];

						Solve3(blas, beta,
							p, r,
							countForDevice, offsetForDevice);
					});
					rr = rrNew;
				}
			}
		}

		/// <summary>
		/// 結果を読み込む
		/// </summary>
		public override void Read()
		{
			// 全デバイスで
			Parallel.For(0, deviceCount, deviceID =>
			{
				SetDevice(deviceID);

				var countForDevice = CountForDevice(deviceID);
				var offsetForDevice = offsetsForDevice[deviceID];

				// 結果を読み込む
				vectorX[deviceID].CopyTo(ref x, countForDevice, offsetForDevice);
			});
		}

		/// <summary>
		/// このデバイスが計算する要素数を取得する
		/// </summary>
		/// <param name="deviceID">デバイス番号</param>
		/// <returns>要素数</returns>
		int CountForDevice(int deviceID)
		{
			// 範囲内なら、先頭位置の差を要素数とする
			return (0 <= deviceID) && (deviceID < deviceCount) ? offsetsForDevice[deviceID + 1] - offsetsForDevice[deviceID] : 0;
		}
	}
}