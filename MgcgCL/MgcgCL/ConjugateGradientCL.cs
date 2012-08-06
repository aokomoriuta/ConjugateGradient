using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 共役勾配法
	/// </summary>
	public class ConjugateGradientCL : LinerEquations
	{
		/// <summary>
		/// 最小繰り返し回数
		/// </summary>
		int minIteration;

		/// <summary>
		/// 最大繰り返し回数
		/// </summary>
		int maxIteration;

		/// <summary>
		/// 収束判定誤差の2乗
		/// </summary>
		double allowableResidual2;


		ComputeCommandQueue queue;

		ComputeKernel setAllVectorKernel;
		ComputeKernel plusEachVectorKernel;
		ComputeKernel multiplyEachVectorKernel;
		ComputeKernel addVectorSecondHalfToFirstHalfKernel;
		ComputeKernel multiplyMatrixVectorKernel;
		ComputeKernel columnVectorToRowKernel;

		// バッファーを作成
		ComputeBuffer<double> bufferA;
		ComputeBuffer<long> bufferAColumnIndeces;
		ComputeBuffer<long> bufferANonzeroCounts;
		ComputeBuffer<double> bufferB;
		ComputeBuffer<double> bufferX;
		ComputeBuffer<double> bufferAp;
		ComputeBuffer<double> bufferP;
		ComputeBuffer<double> bufferR;

		// 計算に使うバッファーを作成
		ComputeBuffer<double> bufferForDot;
		double[] answerForDot;
		ComputeBuffer<double> bufferForMatrix_x_Vector;


		/// <summary>
		/// 共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientCL(int count, long maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount)
		{
			// 最小・最大繰り返し回数を設定
			this.minIteration = _minIteration;
			this.maxIteration = _maxIteration;

			// 収束判定誤差を設定
			this.allowableResidual2 = allowableResidual * allowableResidual;


			// プラットフォームとデバイス群を取得
			var platform = ComputePlatform.Platforms[0];
			var devices = platform.Devices;

			// コンテキストを作成
			var context = new ComputeContext(devices, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// キューを作成
			queue = new ComputeCommandQueue(context, devices[0], ComputeCommandQueueFlags.None);


			// バッファーを作成
			bufferA = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, this.Count * this.A.MaxNonzeroCountPerRow);
			bufferAColumnIndeces = new ComputeBuffer<long>(context, ComputeMemoryFlags.ReadOnly, this.Count * this.A.MaxNonzeroCountPerRow);
			bufferANonzeroCounts = new ComputeBuffer<long>(context, ComputeMemoryFlags.ReadOnly, this.Count);
			bufferB = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, this.Count);
			bufferX = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferAp = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferP = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferR = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);

			// 計算に使うバッファーを作成
			bufferForDot = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			answerForDot = new double[1];
			bufferForMatrix_x_Vector = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count * this.A.MaxNonzeroCountPerRow);

			// データを転送
			queue.WriteToBuffer(this.A.Elements, bufferA, false, null);
			queue.WriteToBuffer(this.b, bufferB, false, null);
			queue.WriteToBuffer(this.A.ColumnIndeces, bufferAColumnIndeces, false, null);
			queue.WriteToBuffer(this.A.NonzeroCounts, bufferANonzeroCounts, false, null);

			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.Mgcg);

			// ビルドしてみて
			try
			{
				program.Build(devices,
					"", null, IntPtr.Zero);
				//"-Werror", null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// 例外を投げる
				throw new ApplicationException(program.GetBuildLog(devices[0]), ex);
			}

			// カーネルを作成
			setAllVectorKernel = program.CreateKernel("SetAllVector");
			plusEachVectorKernel = program.CreateKernel("PlusEachVector");
			multiplyEachVectorKernel = program.CreateKernel("MultiplyEachVector");
			addVectorSecondHalfToFirstHalfKernel = program.CreateKernel("AddVectorSecondHalfToFirstHalf");
			multiplyMatrixVectorKernel = program.CreateKernel("MultiplyMatrixVector");
			columnVectorToRowKernel = program.CreateKernel("ColumnVectorToRow");
		}

		/// <summary>
		/// OpenCLを使って方程式を解く
		/// </summary>
		public void Solve()
		{
			// 未知数ベクトルを0で初期化
			this.InitializeVector(bufferX, 0);

			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = (LDLr)_0
			 */
			this.InitializeVector(bufferAp, 0);
			this.VectorPlusVector(bufferR, bufferB, bufferAp, -1);
			queue.CopyBuffer(bufferR, bufferP, null);


			// 収束したかどうか
			bool converged = false;

			// 収束しない間繰り返す
			for(int iteration = 0; !converged; iteration++)
			{
				// 計算を実行
				/*
				 * rr = r・r
				 * Ap = A * p
				 * α = rr/(p・Ap)
				 * x' += αp
				 * r' -= αAp
				 * r'r' = r'・r'
				 */
				double rr = this.VectorDotVector(bufferR, bufferR);
				this.Matrix_x_Vector(bufferAp, bufferA, bufferAColumnIndeces, bufferANonzeroCounts, bufferP);
				double alpha = rr / this.VectorDotVector(bufferP, bufferAp);
				this.VectorPlusVector(bufferX, bufferX, bufferP, alpha);
				this.VectorPlusVector(bufferR, bufferR, bufferAp, -alpha);
				double rrNew = this.VectorDotVector(bufferR, bufferR);

				// 最小繰り返し回数未満なら
				if(iteration < this.minIteration)
				{
					// 収束していない
					converged = false;
				}
				// 最大繰り返し回数を超えていたら
				else if(iteration > this.maxIteration)
				{
					// 例外
					throw new System.ApplicationException("圧力方程式が収束しませんでした。");
				}
				// それ以外の時
				else
				{
					// 残差ベクトルの大きさが収束判定誤差より小さいかどうかを計算
					converged = (rrNew < this.allowableResidual2);
				}

				// 収束していたら
				if(converged)
				{
					// 計算終了
					break;
				}

				// 残りの計算を実行
				/*
				 * β= r'r'/rLDLr
				 * p = r' + βp
				 */
				double beta = rrNew / rr;
				this.VectorPlusVector(bufferP, bufferR, bufferP, beta);
			}

			// 計算結果を読み込み
			var result = new double[this.Count];
			queue.ReadFromBuffer(bufferX, ref result, false, null);

			// ここまで待機
			queue.Finish();

			// 結果を複製
			result.CopyTo(this.x, 0);
		}

		private void Matrix_x_Vector(ComputeBuffer<double> answer, ComputeBuffer<double> matrix, ComputeBuffer<long> columnIndeces, ComputeBuffer<long> nonzeroCounts, ComputeBuffer<double> vector)
		{
			// 各要素同士の掛け算を実行
			//  # 解を格納するベクトル
			//  # 掛けられる値
			//  # 掛ける値
			multiplyMatrixVectorKernel.SetMemoryArgument(0, bufferForMatrix_x_Vector);
			multiplyMatrixVectorKernel.SetMemoryArgument(1, matrix);
			multiplyMatrixVectorKernel.SetMemoryArgument(2, columnIndeces);
			multiplyMatrixVectorKernel.SetMemoryArgument(3, nonzeroCounts);
			multiplyMatrixVectorKernel.SetMemoryArgument(4, vector);
			queue.Execute(multiplyMatrixVectorKernel, null, new long[] { this.Count, this.A.MaxNonzeroCountPerRow }, null, null);

			// 以前の大きさを設定
			long oldSize = this.A.MaxNonzeroCountPerRow;

			// リダクションの計算が終了するまで書く大きさで
			for(long size = oldSize / 2; size > 0; size /= 2)
			{
				// 前の大きさが奇数だった場合は1つ上の偶数にする
				size += (oldSize % 2 == 1) ? 1 : 0;

				// 後半の値を前半の値に加える（リダクション）
				//  # 対象となる大きさ
				//  # 操作対象のベクトル
				addVectorSecondHalfToFirstHalfKernel.SetValueArgument(0, oldSize);
				addVectorSecondHalfToFirstHalfKernel.SetValueArgument(1, this.A.MaxNonzeroCountPerRow);
				addVectorSecondHalfToFirstHalfKernel.SetMemoryArgument(2, bufferForMatrix_x_Vector);
				queue.Execute(addVectorSecondHalfToFirstHalfKernel, null, new long[] { this.Count, size }, null, null);

				// 今回の大きさを保存
				oldSize = size;
			}

			// 縦ベクトルを横ベクトルに変換して、結果に格納
			//  # 配列
			//  # 行列
			//  # 行列の列数
			columnVectorToRowKernel.SetMemoryArgument(0, answer);
			columnVectorToRowKernel.SetMemoryArgument(1, bufferForMatrix_x_Vector);
			columnVectorToRowKernel.SetValueArgument(2, this.A.MaxNonzeroCountPerRow);
			queue.Execute(columnVectorToRowKernel, null, new long[] { this.Count }, null, null);
		}

		private double VectorDotVector(ComputeBuffer<double> left, ComputeBuffer<double> right)
		{
			// 各要素同士の掛け算を実行
			//  # 解を格納するベクトル
			//  # 掛けられる値
			//  # 掛ける値
			multiplyEachVectorKernel.SetMemoryArgument(0, bufferForDot);
			multiplyEachVectorKernel.SetMemoryArgument(1, left);
			multiplyEachVectorKernel.SetMemoryArgument(2, right);
			queue.Execute(multiplyEachVectorKernel, null, new long[] { this.Count }, null, null);

			// 以前の大きさを設定
			long oldSize = this.Count;

			// リダクションの計算が終了するまで書く大きさで
			for(long size = oldSize / 2; size > 0; size /= 2)
			{
				// 前の大きさが奇数だった場合は大きさを1つ増やす
				size += (oldSize % 2 == 1) ? 1 : 0;

				// 後半の値を前半の値に加える（リダクション）
				//  # 対象となる大きさ
				//  # 操作対象のベクトル
				addVectorSecondHalfToFirstHalfKernel.SetValueArgument(0, oldSize);
				addVectorSecondHalfToFirstHalfKernel.SetValueArgument(1, 1L);
				addVectorSecondHalfToFirstHalfKernel.SetMemoryArgument(2, bufferForDot);
				queue.Execute(addVectorSecondHalfToFirstHalfKernel, null, new long[] { 1, size }, null, null);

				// 今回の大きさを保存
				oldSize = size;
			}

			// 結果を取得
			queue.ReadFromBuffer(bufferForDot, ref answerForDot, true, 0, 0, 1, null);

			// 結果を返す
			return answerForDot[0];
		}

		private void VectorPlusVector(ComputeBuffer<double> answer, ComputeBuffer<double> left, ComputeBuffer<double> right, double C)
		{
			// 各要素同士の足し算を実行
			//  # 解を格納するベクトル
			//  # 足すベクトル
			//  # 足されるベクトル
			//  # 足されるベクトルにかかる係数
			plusEachVectorKernel.SetMemoryArgument(0, answer);
			plusEachVectorKernel.SetMemoryArgument(1, left);
			plusEachVectorKernel.SetMemoryArgument(2, right);
			plusEachVectorKernel.SetValueArgument(3, C);
			queue.Execute(plusEachVectorKernel, null, new long[] { this.Count }, null, null);
		}

		private void InitializeVector(ComputeBuffer<double> vector, double value)
		{
			// 各要素同士の足し算を実行
			//  # 初期化先のベクトル
			//  # 初期値
			setAllVectorKernel.SetMemoryArgument(0, vector);
			setAllVectorKernel.SetValueArgument(1, value);
			queue.Execute(setAllVectorKernel, null, new long[] { this.Count }, null, null);
		}
	}
}