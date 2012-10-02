using Cloo;
using System;
using MathFunctions = System.Math;
namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 1GPUで共役勾配法
	/// </summary>
	public class ConjugateGradientSingleGpu : ConjugateGradient
	{
		/// <summary>
		/// コマンドキュー
		/// </summary>
		readonly ComputeCommandQueue queue;

		#region カーネル
		/// <summary>
		/// ベクトルの各要素を加算する
		/// </summary>
		ComputeKernel addVectorVector;

		/// <summary>
		/// ベクトルとベクトルの各要素の積算をする
		/// </summary>
		ComputeKernel multiplyVectorVector;

		/// <summary>
		/// リダクションで配列の全総和を計算する
		/// </summary>
		ComputeKernel reductionSum;

		/// <summary>
		/// リダクションで配列の最大値を計算する
		/// </summary>
		ComputeKernel reductionMax;

		/// <summary>
		/// 行列とベクトルの積算をする
		/// </summary>
		ComputeKernel matrix_x_Vector;
		#endregion

		#region バッファー
		/// <summary>
		/// 係数行列
		/// </summary>
		ComputeBuffer<double> bufferA;

		/// <summary>
		/// 列番号
		/// </summary>
		ComputeBuffer<int> bufferColumnIndeces;

		/// <summary>
		/// 非ゼロ要素数
		/// </summary>
		ComputeBuffer<int> bufferNonzeroCounts;

		/// <summary>
		/// 右辺ベクトル
		/// </summary>
		ComputeBuffer<double> bufferB;

		/// <summary>
		/// 未知数
		/// </summary>
		ComputeBuffer<double> bufferX;

		/// <summary>
		/// 係数行列と探索方向ベクトルの積
		/// </summary>
		ComputeBuffer<double> bufferAp;

		/// <summary>
		/// 探索方向
		/// </summary>
		ComputeBuffer<double> bufferP;

		/// <summary>
		/// 残差
		/// </summary>
		ComputeBuffer<double> bufferR;


		/// <summary>
		/// ベクトル用バッファー
		/// </summary>
		ComputeBuffer<double> bufferVector;

		/// <summary>
		/// リダクションの答えに使うバッファー
		/// </summary>
		double[] answerForReduction;

		/// <summary>
		/// グローバルアイテム数
		/// </summary>
		readonly long[] globalWorkSize;

		/// <summary>
		/// ローカルアイテム数
		/// </summary>
		readonly long[] localWorkSize;

		/// <summary>
		/// 行列計算で使用するベクトルの前後のバッファーの大きさ
		/// </summary>
		readonly int bufferSizeOfVectorOnMatrixMultiplying;
		#endregion


		/// <summary>
		/// OpenCLでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientSingleGpu(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];

			// コンテキストを作成
			var context = new ComputeContext(ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;
			var device = devices[0];


			globalWorkSize = new long[] { this.Count };
			localWorkSize = new long[] { device.MaxWorkItemSizes[0] };

			// キューを作成
			queue = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None);

			// バッファーを作成
			bufferA = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, this.A.Elements.Length);
			bufferColumnIndeces = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, this.A.ColumnIndeces.Length);
			bufferNonzeroCounts = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, this.A.NonzeroCounts.Length);

			// バッファーを作成
			bufferB = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly, this.b.Length);
			bufferX = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.x.Length);
			bufferAp = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferP = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferR = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);

			// 計算に使うバッファーを作成
			bufferVector = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			answerForReduction = new double[1];

			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.Mgcg);

			// ビルドしてみて
			try
			{
				string realString = "double";

				program.Build(devices,
					string.Format(" -D REAL={0} -D MAX_NONZERO_COUNT={1}", realString, this.A.MaxNonzeroCountPerRow),
					null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// 例外を投げる
				throw new ApplicationException(program.GetBuildLog(devices[0]), ex);
			}

			// カーネルを作成
			addVectorVector = program.CreateKernel("AddVectorVector");
			multiplyVectorVector = program.CreateKernel("MultiplyVectorVector");
			reductionSum = program.CreateKernel("ReductionSum");
			reductionMax = program.CreateKernel("ReductionMaxAbsolute");
			matrix_x_Vector = program.CreateKernel("Matrix_x_Vector");

			// バッファーは最大非ゼロ要素数の半分
			bufferSizeOfVectorOnMatrixMultiplying = this.A.MaxNonzeroCountPerRow / 2;
		}

		/// <summary>
		/// 初期化処理（データの転送など）
		/// </summary>
		public void Initialize()
		{
			//int workGroupCount = (int)MathFunctions.Ceiling((double)globalWorkSize[0] / localWorkSize[0]);

			//for(int workGroupIndex = 0; workGroupIndex < workGroupCount; workGroupIndex++)
			//{
			//	int minI = this.Count;
			//	int maxI = 0;

			//	for(int i = workGroupIndex * (int)localWorkSize[0]; i < (workGroupIndex + 1) * (int)localWorkSize[0]; i++)
			//	{
			//		for(int j = 0; j < this.A.NonzeroCounts[i]; j++)
			//		{
			//			int k = this.A.ColumnIndeces[i * this.A.MaxNonzeroCountPerRow + j];

			//			minI = MathFunctions.Min(minI, k);
			//			maxI = MathFunctions.Max(maxI, k);
			//		}
			//	}

			//	int start  = (int)MathFunctions.Max(0         ,  workGroupIndex *      (int)localWorkSize[0] - 0.5 * this.A.MaxNonzeroCountPerRow);
			//	int finish = (int)MathFunctions.Min(this.Count, (workGroupIndex + 1) * (int)localWorkSize[0] + 0.5 * this.A.MaxNonzeroCountPerRow);
			//}

			// 行列、初期未知数、右辺ベクトルデータを転送
			queue.WriteToBuffer(this.A.Elements, bufferA, false, null);
			queue.WriteToBuffer(this.A.ColumnIndeces, bufferColumnIndeces, false, null);
			queue.WriteToBuffer(this.A.NonzeroCounts, bufferNonzeroCounts, false, null);
			queue.WriteToBuffer(this.x, bufferX, false, null);
			queue.WriteToBuffer(this.b, bufferB, false, null);

			queue.Finish();
		}

		/// <summary>
		/// OpenCLを使って方程式を解く
		/// </summary>
		override public void Solve()
		{
			//for(int i = 0; i < this.MinIteration; i++)
			//{
			//	this.Matrix_x_Vector(bufferAp, bufferA, bufferColumnIndeces, bufferNonzeroCounts, bufferX);
			//	this.Matrix_x_Vector(bufferX, bufferA, bufferColumnIndeces, bufferNonzeroCounts, bufferAp);
			//	//this.VectorPlusVector(bufferX, bufferB, bufferX, -1);
			//	Console.WriteLine(i);
			//}

			//// ここまで待機
			//queue.Finish();
			//return;

			// 初期値を設定
			/*
			 * (Ap)_0 = A * x
			 * r_0 = b - Ap
			 * p_0 = (LDLr)_0
			 */
			this.Matrix_x_Vector(bufferAp, bufferA, bufferColumnIndeces, bufferNonzeroCounts, bufferX);
			this.VectorPlusVector(bufferR, bufferB, bufferAp, -1);
			queue.CopyBuffer(bufferR, bufferP, null);

			// 収束しない間繰り返す
			for(this.Iteration = 0; ; this.Iteration++)
			{
				// 計算を実行
				/*
				 * rr = r・r
				 * Ap = A * p
				 * α = rr/(p・Ap)
				 * x' += αp
				 * r' -= αAp
				 */
				double rr = this.VectorDotVector(bufferR, bufferR);
				this.Matrix_x_Vector(bufferAp, bufferA, bufferColumnIndeces, bufferNonzeroCounts, bufferP);
				double alpha = rr / this.VectorDotVector(bufferP, bufferAp);
				this.VectorPlusVector(bufferX, bufferX, bufferP, alpha);
				this.VectorPlusVector(bufferR, bufferR, bufferAp, -alpha);

				// 収束したかどうかを取得
				this.Residual = this.MaxAbsolute(bufferR);

				// 収束していたら
				if(this.IsConverged)
				{
					// 繰り返し終了
					break;
				}
				// なかったら
				else
				{
					// 残りの計算を実行
					/*
					 * β= r'r'/rLDLr
					 * p = r' + βp
					 * r'r' = r'・r'
					 */
					double rrNew = this.VectorDotVector(bufferR, bufferR);
					double beta = rrNew / rr;
					this.VectorPlusVector(bufferP, bufferR, bufferP, beta);
				}
			}

			// 計算結果を読み込み
			queue.ReadFromBuffer(bufferX, ref this.x, false, null);

			// ここまで待機
			queue.Finish();
		}

		/// <summary>
		/// 結果を読み込む
		/// </summary>
		public void Read()
		{
			// 計算結果を読み込み
			queue.ReadFromBuffer(bufferX, ref this.x, true, null);
		}


		/// <summary>
		/// ベクトル同士の和を計算する
		/// </summary>
		/// <param name="answer">解の代入先</param>
		/// <param name="left">足されるベクトル</param>
		/// <param name="right">足すベクトル</param>
		/// <param name="C">足すベクトルに掛ける係数</param>
		void VectorPlusVector(ComputeBuffer<double> answer, ComputeBuffer<double> left, ComputeBuffer<double> right, double C = 1)
		{
			// 各要素同士の足し算を実行
			//  # 解を格納するベクトル
			//  # 足すベクトル
			//  # 足されるベクトル
			//  # 足されるベクトルにかかる係数
			addVectorVector.SetMemoryArgument(0, answer);
			addVectorVector.SetMemoryArgument(1, left);
			addVectorVector.SetMemoryArgument(2, right);
			addVectorVector.SetValueArgument(3, C);
			queue.Execute(addVectorVector, null, globalWorkSize, null, null);
		}

		/// <summary>
		/// ベクトルの内積を計算する
		/// </summary>
		/// <param name="left">掛けられるベクトル</param>
		/// <param name="right">掛けるベクトル</param>
		/// <returns>内積（要素同士の積の和）</returns>
		double VectorDotVector(ComputeBuffer<double> left, ComputeBuffer<double> right)
		{
			// 各要素同士の掛け算を実行
			//  # 解を格納するベクトル
			//  # 掛けられる値
			//  # 掛ける値
			multiplyVectorVector.SetMemoryArgument(0, bufferVector);
			multiplyVectorVector.SetMemoryArgument(1, left);
			multiplyVectorVector.SetMemoryArgument(2, right);
			queue.Execute(multiplyVectorVector, null, new long[] { this.Count }, null, null);

			// 計算する配列の要素数
			int targetSize = this.Count;

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)(MathFunctions.Ceiling((double)targetSize / 2 / localWorkSize[0]) * localWorkSize[0]);

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionSum.SetMemoryArgument(0, bufferVector);
				reductionSum.SetValueArgument(1, targetSize);
				reductionSum.SetLocalArgument(2, sizeof(double) * localWorkSize[0]);
				queue.Execute(reductionSum, null, new long[] { globalSize }, localWorkSize, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / (int)localWorkSize[0];
			}

			// 結果を読み込み
			queue.ReadFromBuffer(bufferVector, ref answerForReduction, true, 0, 0, 1, null);

			// 結果を返す
			return answerForReduction[0];
		}

		/// <summary>
		/// 行列とベクトルの積を計算する
		/// </summary>
		/// <param name="result">解の代入先</param>
		/// <param name="matrix">行列</param>
		/// <param name="columnIndeces">列番号</param>
		/// <param name="nonzeroCounts">非ゼロ要素数</param>
		/// <param name="vector">ベクトル</param>
		void Matrix_x_Vector(ComputeBuffer<double> result, ComputeBuffer<double> matrix, ComputeBuffer<int> columnIndeces, ComputeBuffer<int> nonzeroCounts, ComputeBuffer<double> vector)
		{
			// グローバルアイテム数を計算
			long globalSize = (long)MathFunctions.Ceiling((double)this.Count / localWorkSize[0]) * localWorkSize[0];

			// バッファーサイズを設定
			int bufferSize = this.A.MaxNonzeroCountPerRow;

			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			matrix_x_Vector.SetValueArgument(0, this.Count);
			matrix_x_Vector.SetMemoryArgument(1, result);
			matrix_x_Vector.SetMemoryArgument(2, matrix);
			matrix_x_Vector.SetMemoryArgument(3, vector);
			matrix_x_Vector.SetMemoryArgument(4, columnIndeces);
			matrix_x_Vector.SetMemoryArgument(5, nonzeroCounts);
			matrix_x_Vector.SetValueArgument(6, bufferSize);
			matrix_x_Vector.SetLocalArgument(7, sizeof(double) * (2 * bufferSize + localWorkSize[0]));
			queue.Execute(matrix_x_Vector, null, new long[] { globalSize }, localWorkSize, null);
		}

		/// <summary>
		/// ベクトルの中で最大絶対値を探す
		/// </summary>
		/// <param name="target">対象のベクトル</param>
		double MaxAbsolute(ComputeBuffer<double> target)
		{
			// 対象のデータを複製
			queue.CopyBuffer(target, bufferVector, null);

			// 計算する配列の要素数
			int targetSize = this.Count;

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)(MathFunctions.Ceiling((double)targetSize / 2 / localWorkSize[0]) * localWorkSize[0]);

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionMax.SetMemoryArgument(0, bufferVector);
				reductionMax.SetValueArgument(1, targetSize);
				reductionMax.SetLocalArgument(2, sizeof(double) * localWorkSize[0]);
				queue.Execute(reductionMax, null, new long[] { globalSize }, localWorkSize, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / (int)localWorkSize[0];
			}

			// 結果を読み込み
			queue.ReadFromBuffer(bufferVector, ref answerForReduction, true, 0, 0, 1, null);

			// 結果を返す
			return answerForReduction[0];
		}
	}
}