using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 共役勾配法
	/// </summary>
	public class ConjugateGradient : LinerEquations
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

		/// <summary>
		///  残差ベクトル
		/// </summary>
		readonly double[] r;

		/// <summary>
		/// 探索方向ベクトル
		/// </summary>
		readonly double[] p;

		/// <summary>
		/// 係数行列と探索方向ベクトルの積
		/// </summary>
		readonly double[] Ap;

		/// <summary>
		/// 共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradient(int count, long maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount)
		{
			// 残差および探索方向と係数行列と探索方向の積を初期化
			this.r = new double[count];
			this.p = new double[count];
			this.Ap = new double[count];

			// 最小・最大繰り返し回数を設定
			this.minIteration = _minIteration;
			this.maxIteration = _maxIteration;

			// 収束判定誤差を設定
			this.allowableResidual2 = allowableResidual * allowableResidual;
		}

		/// <summary>
		/// 方程式を解く
		/// </summary>
		public void Solve()
		{
			// ベクトルをゼロに初期化
			for(int i = 0; i < this.Count; i++)
			{
				this.Ap[i] = 0;
				this.r[i] = 0;
			}

			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = (LDLr)_0
			 */
			this.A.Multiply(this.Ap, this.x, this.isEnabled);
			this.r.SetAdded(this.b, this.Ap, -1.0);
			this.r.CopyTo(this.p, 0);

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
				double rr = this.r.Dot(this.r);
				this.A.Multiply(this.Ap, this.p, this.isEnabled);
				double alpha = rr / this.p.Dot(this.Ap);
				this.x.SetAdded(this.x, this.p, alpha);
				this.r.SetAdded(this.r, this.Ap, -alpha);
				double rrNew = this.r.Dot(this.r);

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
				this.p.SetAdded(this.r, this.p, beta);
			}
		}

		/// <summary>
		/// OpenCLを使って方程式を解く
		/// </summary>
		public void SolveCL()
		{
			// プラットフォームとデバイス群を取得
			var platform = ComputePlatform.Platforms[0];
			var devices = platform.Devices;

			// コンテキストを作成
			var context = new ComputeContext(devices, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// キューを作成
			var queue = new ComputeCommandQueue(context, devices[0], ComputeCommandQueueFlags.None);


			// バッファーを作成
			var bufferA = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, this.Count * this.A.MaxNonzeroCountPerRow);
			var bufferAColumnIndeces = new ComputeBuffer<long>(context, ComputeMemoryFlags.ReadOnly, this.Count * this.A.MaxNonzeroCountPerRow);
			var bufferANonzeroCounts = new ComputeBuffer<long>(context, ComputeMemoryFlags.ReadOnly, this.Count);
			var bufferB = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, this.Count);
			var bufferX = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			var bufferAp = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			var bufferP = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			var bufferR = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);

			// 
			var bufferForDot = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			var answerForDot = new double[1];
			var bufferForMatrix_x_Vector = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count * this.A.MaxNonzeroCountPerRow);

			// データを転送
			queue.WriteToBuffer(this.A.Elements, bufferA, false, null);
			queue.WriteToBuffer(this.b, bufferB, false, null);
			
			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.Mgcg);

			// ビルドしてみて
			try
			{
				program.Build(devices, "-Werror", null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// 例外を投げる
				throw new ApplicationException(program.GetBuildLog(devices[0]), ex);
			}

			// カーネルを作成
			var setAllVectorKernel = program.CreateKernel("SetAllVector");
			var plusEachVectorKernel = program.CreateKernel("PlusEachVector");
			var multiplyEachVectorKernel = program.CreateKernel("MultiplyEachVector");
			var addVectorSecondHalfToFirstHalfKernel = program.CreateKernel("AddVectorSecondHalfToFirstHalf");
			var multiplyMatrixVectorKernel = program.CreateKernel("MultiplyMatrixVector");
			
			// ベクトルを初期化する処理
			Action<ComputeBuffer<double>, double> initializeVector = (vector, value) =>
			{
				// 各要素同士の足し算を実行
				//  # 初期化先のベクトル
				//  # 初期値
				setAllVectorKernel.SetMemoryArgument(0, vector);
				setAllVectorKernel.SetValueArgument(1, value);
				queue.Execute(setAllVectorKernel, null, new long[] { this.Count }, null, null);
			};

			// ベクトルの足し算を行う処理
			Action<ComputeBuffer<double>, ComputeBuffer<double>, ComputeBuffer<double>, double> vectorPlusVector = (answer, left, right, C)=>
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
			};

			// ベクトルの内積を計算する処理
			Func<ComputeBuffer<double>, ComputeBuffer<double>, double> vectorDotVector = (left, right) =>
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
				for (long size = this.Count / 2; size > 0; size /= 2)
				{
					// 前の大きさが奇数だった場合は大きさを1つ増やす
					size += (oldSize % 2 == 1) ? 1 : 0;

					// 後半の値を前半の値に加える（リダクション）
					//  # 対象となる大きさ
					//  # 操作対象のベクトル
					addVectorSecondHalfToFirstHalfKernel.SetValueArgument(0, oldSize);
					addVectorSecondHalfToFirstHalfKernel.SetMemoryArgument(1, bufferForDot);
					queue.Execute(addVectorSecondHalfToFirstHalfKernel, null, new long[] { size }, null, null);

					// 今回の大きさを保存
					oldSize = size;
				}

				// 結果を取得
				queue.ReadFromBuffer(bufferForDot, ref answerForDot, true, null);

				// 結果を返す
				return answerForDot[0];
			};

			// ベクトルの足し算を行う処理
			Action<ComputeBuffer<double>, ComputeBuffer<double>, ComputeBuffer<long>, ComputeBuffer<long>, ComputeBuffer<double>> matrix_x_Vector = (answer, matrix, columnIndeces, nonzeroCounts, vector)=>
			{
				// 各要素同士の掛け算を実行
				//  # 解を格納するベクトル
				//  # 掛けられる値
				//  # 掛ける値
				multiplyMatrixVectorKernel.SetMemoryArgument(0, bufferForMatrix_x_Vector);
				var s = bufferForMatrix_x_Vector.Count;
				//multiplyMatrixVectorKernel.SetMemoryArgument(1, matrix);
				//multiplyMatrixVectorKernel.SetMemoryArgument(2, columnIndeces);
				//multiplyMatrixVectorKernel.SetMemoryArgument(3, nonzeroCounts);
				//multiplyMatrixVectorKernel.SetMemoryArgument(4, vector);
				queue.Execute(multiplyMatrixVectorKernel, null, new long[] { this.Count, this.A.MaxNonzeroCountPerRow }, null, null);


				var debug = new double[this.Count * this.A.MaxNonzeroCountPerRow];
				queue.ReadFromBuffer(bufferForMatrix_x_Vector, ref debug, true, null);

				int a = 0;
			};

			// 未知数ベクトルを0で初期化
			initializeVector(bufferX, 0);

			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = (LDLr)_0
			 */
			initializeVector(bufferAp, 0);
			vectorPlusVector(bufferR, bufferB, bufferAp, -1);
			queue.CopyBuffer(bufferR, bufferP, null);


			// 計算を実行
			/*
			 * rr = r・r
			 * Ap = A * p
			 * α = rr/(p・Ap)
			 * x' += αp
			 * r' -= αAp
			 * r'r' = r'・r'
			 */
			double rr = vectorDotVector(bufferR, bufferR);
			matrix_x_Vector(bufferAp, bufferA, bufferAColumnIndeces, bufferANonzeroCounts, bufferP);

			// 計算結果を読み込み
			var result = new double[this.Count];
			queue.ReadFromBuffer(bufferAp, ref result, false, null);

			// ここまで待機
			queue.Finish();

			// 結果を複製
			result.CopyTo(this.x, 0);
		}
	}
}