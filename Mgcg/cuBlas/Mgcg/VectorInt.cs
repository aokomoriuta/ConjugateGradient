using System;
using System.Runtime.InteropServices;
namespace LWisteria.Mgcg
{
	/// <summary>
	/// int型のベクトル
	/// </summary>
	public class VectorInt : IDisposable
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
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint="Create_Int")]
		static extern IntPtr Create(int size);

		/// <summary>
		/// ベクトルを廃棄する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Delete_Int")]
		static extern void Delete(IntPtr vec);

		/// <summary>
		/// 配列からデータを複製する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyFromArray_Int")]
		static extern void CopyFrom(IntPtr destination, int[] source,
		int count, int sourceOffset, int destinationOffset);

		/// <summary>
		/// 配列へデータを複製する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyToArray_Int")]
		static extern void CopyTo(IntPtr source, int[] destination,
		int count, int sourceOffset, int destinationOffset);

		/// <summary>
		/// CUDA用ポインタへ変換する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "ToRawPtr_Int")]
		static extern IntPtr ToRawPtr(IntPtr vec);
		#endregion

		/// <summary>
		/// double型のthrustベクトルを生成する
		/// </summary>
		/// <param name="size"></param>
		/// <param name="deviceID"></param>
		public VectorInt(int size)
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
		public void CopyFrom(int[] array, int count, int arrayOffset = 0, int vectorOffset = 0)
		{
			CopyFrom(Ptr, array, count, arrayOffset, vectorOffset);
		}

		/// <summary>
		/// 配列へデータを複製する
		/// </summary>
		/// <param name="array">複製先</param>
		/// <param name="count">複製する要素数</param>
		public void CopyTo(ref int[] array, int count, int arrayOffset = 0, int vectorOffset = 0)
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