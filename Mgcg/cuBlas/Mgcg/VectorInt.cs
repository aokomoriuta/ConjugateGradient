using System;
using System.Runtime.InteropServices;
namespace LWisteria.Mgcg
{
	/// <summary>
	/// int型のベクトル
	/// </summary>
	public class VectorInt
	{
		/// <summary>
		/// ベクトル本体のポインタ
		/// </summary>
		public readonly IntPtr Ptr;

		/// <summary>
		/// 使用するデバイス
		/// </summary>
		public readonly int Device;

		#region バインディング
		/// <summary>
		/// ベクトルを作成する
		/// </summary>
		/// <param name="size"></param>
		/// <param name="deviceID"></param>
		/// <returns></returns>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint="Create_Int")]
		static extern IntPtr Create(int size, int deviceID);

		/// <summary>
		/// ベクトルを廃棄する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "Delete_Int")]
		static extern void Delete(IntPtr vec, int deviceID);

		/// <summary>
		/// 配列からデータを複製する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyFrom_Int")]
		static extern void CopyFrom(IntPtr destination, int[] source, int count, int deviceID);

		/// <summary>
		/// 配列へデータを複製する
		/// </summary>
		/// <param name="vec"></param>
		/// <param name="deviceID"></param>
		[DllImport(MgcgGpu.DLL_NAME, EntryPoint = "CopyTo_Int")]
		static extern void CopyTo(IntPtr source, int[] destination, int count, int deviceID);

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
		public VectorInt(int size, int deviceID = 0)
		{
			Device = deviceID;
			Ptr = Create(size, deviceID);
		}

		/// <summary>
		/// 廃棄する
		/// </summary>
		~VectorInt()
		{
			Delete(Ptr, Device);
		}

		/// <summary>
		/// 配列からデータを複製する
		/// </summary>
		/// <param name="array">複製元</param>
		/// <param name="count">複製する要素数</param>
		public void CopyFrom(int[] array, int count)
		{
			CopyFrom(Ptr, array, count, Device);
		}

		/// <summary>
		/// 配列へデータを複製する
		/// </summary>
		/// <param name="array">複製先</param>
		/// <param name="count">複製する要素数</param>
		public void CopyTo(ref int[] array, int count)
		{
			CopyTo(Ptr, array, count, Device);
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