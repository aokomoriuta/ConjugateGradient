using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace LWisteria.Mgcg
{
	/// <summary>
	/// GPUで共役勾配法
	/// </summary>
	public abstract class ConjugateGradientGpu : ConjugateGradient
	{
		#region バインディング
		/// <summary>
		/// デバイスを設定する
		/// </summary>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "SetDevice")]
		static protected extern void SetDevice(int deviceID);

		/// <summary>
		/// cublasを作成する
		/// </summary>
		/// <param name="deviceID"></param>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CreateBlas")]
		static protected extern IntPtr CreateBlas();

		/// <summary>
		/// cublasを廃棄する
		/// </summary>
		/// <param name="cublas"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "DestroyBlas")]
		static protected extern void DestroyBlas(IntPtr cublas);

		/// <summary>
		/// cusparseを作成する
		/// </summary>
		/// <param name="deviceID"></param>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CreateSparse")]
		static protected extern IntPtr CreateSparse();

		/// <summary>
		/// cusparseを廃棄する
		/// </summary>
		/// <param name="cusparse"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "DestroySparse")]
		static protected extern void DestroySparse(IntPtr cusparse);

		/// <summary>
		/// matDescrを作成する
		/// </summary>
		/// <param name="deviceID"></param>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CreateMatDescr")]
		static protected extern IntPtr CreateMatDescr();

		/// <summary>
		/// matDescrを廃棄する
		/// </summary>
		/// <param name="cusparse"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "DestroyMatDescr")]
		static protected extern void DestroyMatDescr(IntPtr matDescr);
		#endregion

		/// <summary>
		/// GPUでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientGpu(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{}

		/// <summary>
		/// 初期化処理（データの転送など）
		/// </summary>
		abstract public void Initialize();

		/// <summary>
		/// 結果を読み込む
		/// </summary>
		abstract public void Read();
	}
}