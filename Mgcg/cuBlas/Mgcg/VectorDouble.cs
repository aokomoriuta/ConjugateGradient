using System;
using System.Runtime.InteropServices;
namespace LWisteria.Mgcg
{
	/// <summary>
	/// double型のベクトル
	/// </summary>
	public class VectorDouble : IDisposable
	{
		/// <summary>
		/// ベクトル本体のポインタ
		/// </summary>
		public readonly IntPtr Ptr;

		#region バインディング
		/// <summary>
		/// ベクトルを作成する
		/// </summary>
		/// <param name="size"></param>
		/// <param name="deviceID"></param>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint="Create_Double")]
		static extern IntPtr Create(int size);

		/// <summary>
		/// ベクトルを廃棄する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Delete_Double")]
		static extern void Delete(IntPtr vec);

		/// <summary>
		/// 配列からデータを複製する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyFromArray_Double")]
		static extern void CopyFrom(IntPtr destination, double[] source,
			int count, int sourceOffset, int destinationOffset);

		/// <summary>
		/// 配列へデータを複製する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyToArray_Double")]
		static extern void CopyTo(IntPtr source, double[] destination,
			int count, int sourceOffset, int destinationOffset);

		/// <summary>
		/// CUDA用ポインタへ変換する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "ToRawPtr_Double")]
		static extern IntPtr ToRawPtr(IntPtr vec);

		/// <summary>
		/// デバイスへデータを転送する
		/// </summary>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyFromVector_Double")]
		public static extern IntPtr CopyFrom(IntPtr source, IntPtr destination, 
			int count, int sourceOffset, int destinationOffset);
		#endregion

		/// <summary>
		/// double型のthrustベクトルを生成する
		/// </summary>
		/// <param name="size"></param>
		/// <param name="deviceID"></param>
		public VectorDouble(int size)
		{
			Ptr = Create(size);
		}

		/// <summary>
		/// 廃棄する
		/// </summary>
		public void Dispose()
		{
			Delete(Ptr);
		}

		/// <summary>
		/// 配列からデータを複製する
		/// </summary>
		/// <param name="array">複製元</param>
		/// <param name="count">複製する要素数</param>
		public void CopyFrom(double[] array, int count, int arrayOffset = 0, int vectorOffset = 0)
		{
			CopyFrom(Ptr, array, count, arrayOffset, vectorOffset);
		}

		/// <summary>
		/// 配列へデータを複製する
		/// </summary>
		/// <param name="array">複製先</param>
		/// <param name="count">複製する要素数</param>
		public void CopyTo(ref double[] array, int count, int arrayOffset = 0, int vectorOffset = 0)
		{
			CopyTo(Ptr, array, count, vectorOffset, arrayOffset);
		}

		/// <summary>
		/// CUDA用ポインタへ変換する
		/// </summary>
		/// <returns>CUDAメモリのポインタ</returns>
		public IntPtr ToRawPtr()
		{
			return ToRawPtr(Ptr);
		}
	}
}