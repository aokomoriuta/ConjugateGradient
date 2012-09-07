using System;
using Cloo;
using System.Threading.Tasks;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 複数GPUで共役勾配法
	/// </summary>
	public class ConjugateGradientCLParallel : ConjugateGradient
	{
		/// <summary>
		/// 内積で使うワークグループ内ワークアイテム数
		/// </summary>
		readonly int localSize;

		/// <summary>
		/// コマンドキュー
		/// </summary>
		readonly ComputeCommandQueue[] queues;

		/// <summary>
		/// 各デバイスで計算する要素数
		/// </summary>
		static int[] countPerDevice;

		/// <summary>
		/// デバイスが計算する要素の開始地点
		/// </summary>
		static int[] offset;

		#region カーネル
		/// <summary>
		/// ベクトルの各要素を加算する
		/// </summary>
		ComputeKernel[] addVectorVector;

		/// <summary>
		/// ベクトルとベクトルの各要素の積算をする
		/// </summary>
		ComputeKernel[] multiplyVectorVector;

		/// <summary>
		/// リダクションで配列の全総和を計算する
		/// </summary>
		ComputeKernel[] reductionSum;

		/// <summary>
		/// リダクションで配列の最大値を計算する
		/// </summary>
		ComputeKernel[] reductionMax;

		/// <summary>
		/// 行列とベクトルの積算をする
		/// </summary>
		ComputeKernel[] matrix_x_Vector;
		#endregion

		#region バッファー
		/// <summary>
		/// 係数行列
		/// </summary>
		ComputeBuffer<double>[] buffersA;

		/// <summary>
		/// 列番号
		/// </summary>
		ComputeBuffer<int>[] buffersColumnIndeces;
		
		/// <summary>
		/// 非ゼロ要素数
		/// </summary>
		ComputeBuffer<int>[] buffersNonzeroCounts;

		/// <summary>
		/// 右辺ベクトル
		/// </summary>
		ComputeBuffer<double>[] buffersB;

		/// <summary>
		/// 未知数
		/// </summary>
		ComputeBuffer<double>[] buffersX;

		/// <summary>
		/// 係数行列と探索方向ベクトルの積
		/// </summary>
		ComputeBuffer<double>[] buffersAp;

		/// <summary>
		/// 探索方向
		/// </summary>
		ComputeBuffer<double>[] buffersP;

		/// <summary>
		/// 残差
		/// </summary>
		ComputeBuffer<double>[] buffersR;


		/// <summary>
		/// 内積計算に使うバッファー
		/// </summary>
		ComputeBuffer<double>[] buffersForDot;

		/// <summary>
		/// 最大値の算出に使うバッファー
		/// </summary>
		ComputeBuffer<double>[] buffersForMax;

		/// <summary>
		/// 行列とベクトルの積で使用する全ベクトルのバッファー
		/// </summary>
		ComputeBuffer<double>[] bufferAllVector;

		/// <summary>
		/// 行列とベクトルの積で使用する全ベクトル
		/// </summary>
		double[] allVector;

		/// <summary>
		/// リダクションの答えに使うバッファー
		/// </summary>
		double[] answerForReduction;
		#endregion


		/// <summary>
		/// OpenCLでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientCLParallel(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];

			// コンテキストを作成
			var context = new ComputeContext(Cloo.ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;
			
			// 内積の計算の場合は、回せる最大の数
			this.localSize = (int)devices[0].MaxWorkGroupSize;


			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.Mgcg);

			// ビルドしてみて
			try
			{
				string realString = "double";

				program.Build(devices,
					string.Format(" -D REAL={0} -D MAX_NONZERO_COUNT={1} -Werror", realString, this.A.MaxNonzeroCountPerRow),
					null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// 例外を投げる
				throw new ApplicationException(program.GetBuildLog(devices[0]), ex);
			}


			// 各デバイスで計算する要素数を初期化
			countPerDevice = new int[devices.Count];

			// 1デバイスが計算する最大要素数を計算
			int maxCountPerDevice = (int)Math.Ceiling((double)this.Count / devices.Count);

			// デバイスの計算開始番号を作成
			offset = new int[devices.Count];
			
			// キュー配列を作成
			queues = new ComputeCommandQueue[devices.Count];

			// カーネル配列を作成
			addVectorVector = new ComputeKernel[devices.Count];
			multiplyVectorVector = new ComputeKernel[devices.Count];
			reductionSum = new ComputeKernel[devices.Count];
			reductionMax = new ComputeKernel[devices.Count];
			matrix_x_Vector = new ComputeKernel[devices.Count];

			// バッファー配列を作成
			buffersA = new ComputeBuffer<double>[devices.Count];
			buffersColumnIndeces = new ComputeBuffer<int>[devices.Count];
			buffersNonzeroCounts = new ComputeBuffer<int>[devices.Count];
			buffersB = new ComputeBuffer<double>[devices.Count];
			buffersX = new ComputeBuffer<double>[devices.Count];
			buffersAp = new ComputeBuffer<double>[devices.Count];
			buffersP = new ComputeBuffer<double>[devices.Count];
			buffersR = new ComputeBuffer<double>[devices.Count];
			buffersForDot = new ComputeBuffer<double>[devices.Count];
			buffersForMax = new ComputeBuffer<double>[devices.Count];
			bufferAllVector = new ComputeBuffer<double>[devices.Count];
			answerForReduction = new double[devices.Count];
			allVector = new double[this.Count];

			// 全デバイスについて
			for(int i = 0; i < devices.Count; i++)
			{
				// 計算する要素数を計算
				countPerDevice[i] = maxCountPerDevice - ((i < maxCountPerDevice * devices.Count - this.Count) ? 1 : 0);

				// 計算開始番号を設定
				offset[i] = (i == 0) ? 0 : (offset[i - 1] + countPerDevice[i - 1]);


				// キューを作成
				queues[i] = new ComputeCommandQueue(context, devices[i], ComputeCommandQueueFlags.None);

				// カーネルを作成
				addVectorVector[i] = program.CreateKernel("AddVectorVector");
				multiplyVectorVector[i] = program.CreateKernel("MultiplyVectorVector");
				reductionSum[i] = program.CreateKernel("ReductionSum");
				reductionMax[i] = program.CreateKernel("ReductionMaxAbsolute");
				matrix_x_Vector[i] = program.CreateKernel("Matrix_x_Vector");

				// 行列のバッファーを作成
				buffersA[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i] * this.A.MaxNonzeroCountPerRow);
				buffersColumnIndeces[i] = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i] * this.A.MaxNonzeroCountPerRow);
				buffersNonzeroCounts[i] = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i]);

				// 右辺ベクトル、未知数、探索方向、残差、行列と探索方向の積のバッファーを作成
				buffersB[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i]);
				buffersX[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, countPerDevice[i]);
				buffersAp[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, countPerDevice[i]);
				buffersP[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, countPerDevice[i]);
				buffersR[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, countPerDevice[i]);

				// 計算に使用するバッファーの作成
				buffersForDot[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, countPerDevice[i]);
				buffersForMax[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, countPerDevice[i]);
				bufferAllVector[i] = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			}
		}

		public void Initialize()
		{
			// 全キューについて
			Parallel.For(0, queues.Length, (i) =>
			{
				// 行列、初期未知数、右辺ベクトルデータを転送
				queues[i].WriteToBuffer(this.A.Elements, buffersA[i], false, offset[i] * this.A.MaxNonzeroCountPerRow, 0, countPerDevice[i] * this.A.MaxNonzeroCountPerRow, null);
				queues[i].WriteToBuffer(this.A.ColumnIndeces, buffersColumnIndeces[i], false, offset[i] * this.A.MaxNonzeroCountPerRow, 0, countPerDevice[i] * this.A.MaxNonzeroCountPerRow, null);
				queues[i].WriteToBuffer(this.A.NonzeroCounts, buffersNonzeroCounts[i], false, offset[i], 0, countPerDevice[i], null);
				queues[i].WriteToBuffer(this.x, buffersX[i], false, offset[i], 0, countPerDevice[i], null);
				queues[i].WriteToBuffer(this.b, buffersB[i], false, offset[i], 0, countPerDevice[i], null);
			});
		}

		/// <summary>
		/// OpenCLを使って方程式を解く
		/// </summary>
		override public void Solve()
		{
			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = r_0
			 */
			this.Matrix_x_Vector(buffersX, buffersA, buffersColumnIndeces, buffersNonzeroCounts, buffersX);
			this.VectorPlusVector(buffersR, buffersB, buffersX, -1);
			Parallel.For(0, queues.Length, (i) =>
			{
				queues[i].CopyBuffer(buffersR[i], buffersP[i], null);
			});

			// 収束したかどうか
			bool converged = false;

			// 収束しない間繰り返す
			for(this.Iteration = 0; !converged; this.Iteration++)
			{
				// 計算を実行
				/*
				 * rr = r・r
				 * Ap = A * p
				 * α = rr/(p・Ap)
				 * x' += αp
				 * r' -= αAp
				 */
				double rr = this.VectorDotVector(buffersR, buffersR);
				this.Matrix_x_Vector(buffersAp, buffersA, buffersColumnIndeces, buffersNonzeroCounts, buffersP);
				double alpha = rr / this.VectorDotVector(buffersP, buffersAp);
				this.VectorPlusVector(buffersX, buffersX, buffersP, alpha);
				this.VectorPlusVector(buffersR, buffersR, buffersAp, -alpha);

				// 収束したかどうかを取得
				converged = this.IsConverged(this.MaxAbsolute(buffersR));

				// 収束していなかったら
				if(!converged)
				{
					// 残りの計算を実行
					/*
					 * β= r'r'/rLDLr
					 * p = r' + βp
					 * r'r' = r'・r'
					 */
					double rrNew = this.VectorDotVector(buffersR, buffersR);
					double beta = rrNew / rr;
					this.VectorPlusVector(buffersP, buffersR, buffersP, beta);
				}
			}

			// 全キューについて
			Parallel.For(0, queues.Length, (i) =>
			{
				// 計算結果を読み込み
				queues[i].ReadFromBuffer(buffersX[i], ref this.x, false, 0, offset[i], countPerDevice[i], null);

				// ここまで待機
				queues[i].Finish();
			});
		}

		/// <summary>
		/// ベクトル同士の和を計算する
		/// </summary>
		/// <param name="answer">解の代入先</param>
		/// <param name="left">足されるベクトル</param>
		/// <param name="right">足すベクトル</param>
		/// <param name="C">足すベクトルに掛ける係数</param>
		void VectorPlusVector(ComputeBuffer<double>[] answer, ComputeBuffer<double>[] left, ComputeBuffer<double>[] right, double C = 1)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 各要素同士の足し算を実行
				//  # 解を格納するベクトル
				//  # 足すベクトル
				//  # 足されるベクトル
				//  # 足されるベクトルにかかる係数
				addVectorVector[i].SetMemoryArgument(0, answer[i]);
				addVectorVector[i].SetMemoryArgument(1, left[i]);
				addVectorVector[i].SetMemoryArgument(2, right[i]);
				addVectorVector[i].SetValueArgument(3, C);
				queues[i].Execute(addVectorVector[i], null, new long[] { countPerDevice[i] }, null, null);
			});
		}


		/// <summary>
		/// ベクトルの内積を計算する
		/// </summary>
		/// <param name="left">掛けられるベクトル</param>
		/// <param name="right">掛けるベクトル</param>
		/// <returns>内積（要素同士の積の和）</returns>
		double VectorDotVector(ComputeBuffer<double>[] left, ComputeBuffer<double>[] right)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 各要素同士の掛け算を実行
				//  # 解を格納するベクトル
				//  # 掛けられる値
				//  # 掛ける値
				multiplyVectorVector[i].SetMemoryArgument(0, buffersForDot[i]);
				multiplyVectorVector[i].SetMemoryArgument(1, left[i]);
				multiplyVectorVector[i].SetMemoryArgument(2, right[i]);
				queues[i].Execute(multiplyVectorVector[i], null, new long[] { countPerDevice[i] }, null, null);

				// 計算する配列の要素数
				int targetSize = countPerDevice[i];

				// 計算する配列の要素数が1以上の間
				while(targetSize > 1)
				{
					// ワークアイテム数を計算
					int globalSize = (int)Math.Ceiling((double)targetSize / 2 / localSize) * localSize;

					// 隣との和を計算
					//  # 和を計算するベクトル
					//  # 計算する要素数
					//  # ローカルメモリ
					reductionSum[i].SetMemoryArgument(0, buffersForDot[i]);
					reductionSum[i].SetValueArgument(1, targetSize);
					reductionSum[i].SetLocalArgument(2, sizeof(double) * localSize);
					queues[i].Execute(reductionSum[i], null, new long[] { globalSize }, new long[] { localSize }, null);

					// 次の配列の要素数を今のワークアイテム数にする
					targetSize = globalSize / localSize;
				}

				// 結果を読み込み
				queues[i].ReadFromBuffer(buffersForDot[i], ref answerForReduction, true, 0, i, 1, null);
			});

			// 全キューについて
			for(int i = 1; i < queues.Length; i++)
			{
				// 他のデバイスの結果を合計する
				answerForReduction[0] += answerForReduction[i];
			}

			// 結果を返す
			return answerForReduction[0];
		}

		/// <summary>
		/// 行列とベクトルの積を計算する
		/// </summary>
		/// <param name="answer">解の代入先</param>
		/// <param name="matrix">行列</param>
		/// <param name="columnIndeces">列番号</param>
		/// <param name="nonzeroCounts">非ゼロ要素数</param>
		/// <param name="vector">ベクトル</param>
		void Matrix_x_Vector(ComputeBuffer<double>[] answer, ComputeBuffer<double>[] matrix, ComputeBuffer<int>[] columnIndeces, ComputeBuffer<int>[] nonzeroCounts, ComputeBuffer<double>[] vector)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// ベクトルデータを複製
				queues[i].CopyBuffer(vector[i], bufferAllVector[i], 0, offset[i], countPerDevice[i], null);

				// 1つ前のキューから
				if(i > 0)
				{
					// 境界データを受け取って自分のデータに格納
					queues[i - 1].ReadFromBuffer(vector[i - 1], ref allVector, true, countPerDevice[i - 1] - this.A.MaxNonzeroCountPerRow, offset[i] - this.A.MaxNonzeroCountPerRow, this.A.MaxNonzeroCountPerRow, null);
					queues[i].WriteToBuffer(allVector, bufferAllVector[i], false, offset[i] - this.A.MaxNonzeroCountPerRow, offset[i] - this.A.MaxNonzeroCountPerRow, this.A.MaxNonzeroCountPerRow, null);
				}
				// 1つ後ろのキューから
				if(i < queues.Length - 1)
				{
					// 境界データを受け取って自分のデータに格納
					queues[i + 1].ReadFromBuffer(vector[i + 1], ref allVector, true, 0, offset[i + 1], this.A.MaxNonzeroCountPerRow, null);
					queues[i].WriteToBuffer(allVector, bufferAllVector[i], false, offset[i + 1], offset[i + 1], this.A.MaxNonzeroCountPerRow, null);
				}

				// 行列とベクトルの積を実行
				//  # 解を格納するベクトル
				//  # 行列
				//  # ベクトル
				//  # 列番号
				//  # 非ゼロ要素数
				matrix_x_Vector[i].SetMemoryArgument(0, answer[i]);
				matrix_x_Vector[i].SetMemoryArgument(1, matrix[i]);
				matrix_x_Vector[i].SetMemoryArgument(2, bufferAllVector[i]);
				matrix_x_Vector[i].SetMemoryArgument(3, columnIndeces[i]);
				matrix_x_Vector[i].SetMemoryArgument(4, nonzeroCounts[i]);
				queues[i].Execute(matrix_x_Vector[i], null, new long[] { countPerDevice[i] }, null, null);
			});
		}

		/// <summary>
		/// ベクトルの中で最大絶対値を探す
		/// </summary>
		/// <param name="target">対象のベクトル</param>
		double MaxAbsolute(ComputeBuffer<double>[] target)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 対象のデータを複製
				queues[i].CopyBuffer(target[i], buffersForMax[i], null);

				// 計算する配列の要素数
				int targetSize = countPerDevice[i];

				// 計算する配列の要素数が1以上の間
				while(targetSize > 1)
				{
					// ワークアイテム数を計算
					int globalSize = (int)Math.Ceiling((double)targetSize / 2 / localSize) * localSize;

					// 隣との和を計算
					//  # 和を計算するベクトル
					//  # 計算する要素数
					//  # ローカルメモリ
					reductionMax[i].SetMemoryArgument(0, buffersForMax[i]);
					reductionMax[i].SetValueArgument(1, targetSize);
					reductionMax[i].SetLocalArgument(2, sizeof(double) * localSize);
					queues[i].Execute(reductionMax[i], null, new long[] { globalSize }, new long[] { localSize }, null);

					// 次の配列の要素数を今のワークアイテム数にする
					targetSize = globalSize / localSize;
				}

				// 結果を読み込み
				queues[i].ReadFromBuffer(buffersForMax[i], ref answerForReduction, true, 0, i, 1, null);
			});

			// 全キューについて
			for(int i = 1; i < queues.Length; i++)
			{
				// 他のデバイスの結果と比較して大きい方を格納する
				answerForReduction[0] = Math.Max(answerForReduction[0], answerForReduction[i]);
			}

			// 結果を返す
			return answerForReduction[0];
		}
	}
}