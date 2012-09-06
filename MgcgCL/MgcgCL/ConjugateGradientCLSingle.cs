using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;
using System.Collections.Generic;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 1GPUで共役勾配法
	/// </summary>
	public class ConjugateGradientCLSingle : ConjugateGradient
	{
		/// <summary>
		/// 内積で使うワークグループ内ワークアイテム数
		/// </summary>
		readonly int localSize;

		/// <summary>
		/// コマンドキュー
		/// </summary>
		readonly ComputeCommandQueue queue;

		#region カーネル
		/// <summary>
		/// ベクトルの要素を設定する
		/// </summary>
		ComputeKernel setAllVector;

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
		/// 生成項
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
		/// 内積計算に使うバッファー
		/// </summary>
		ComputeBuffer<double> bufferForDot;

		/// <summary>
		/// リダクションの答えに使うバッファー
		/// </summary>
		double[] answerForReduction;

		/// <summary>
		/// 行列とベクトルの積の計算に使うバッファー
		/// </summary>
		ComputeBuffer<double> bufferForMatrix_x_Vector;


		/// <summary>
		/// 最大値の算出に使うバッファー
		/// </summary>
		ComputeBuffer<double> bufferForMax;
		#endregion


		/// <summary>
		/// OpenCLでの共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientCLSingle(int count, int maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];

			// コンテキストを作成
			var context = new ComputeContext(Cloo.ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;

			
			// キューを作成
			queue = new ComputeCommandQueue(context, devices[0], ComputeCommandQueueFlags.None);

			// バッファーを作成
			bufferA = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.A.Elements);
			bufferColumnIndeces = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.A.ColumnIndeces);
			bufferNonzeroCounts = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.A.NonzeroCounts);

			// バッファーを作成
			bufferB = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.b);
			bufferX = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, this.x);
			bufferAp = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferP = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			bufferR = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);

			// 計算に使うバッファーを作成
			bufferForDot = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);
			answerForReduction = new double[1];
			bufferForMatrix_x_Vector = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count * this.A.MaxNonzeroCountPerRow);
			bufferForMax = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);

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

			// カーネルを作成
			setAllVector = program.CreateKernel("SetAllVector");
			addVectorVector = program.CreateKernel("AddVectorVector");
			multiplyVectorVector = program.CreateKernel("MultiplyVectorVector");
			reductionSum = program.CreateKernel("ReductionSum");
			reductionMax = program.CreateKernel("ReductionMaxAbsolute");
			matrix_x_Vector = program.CreateKernel("Matrix_x_Vector");

			// 内積の計算の場合は、回せる最大の数
			this.localSize = (int)devices[0].MaxWorkGroupSize;
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
			 * p_0 = (LDLr)_0
			 */
			this.Matrix_x_Vector(bufferAp, bufferA, bufferColumnIndeces, bufferNonzeroCounts, bufferX);
			this.VectorPlusVector(bufferR, bufferB, bufferAp, -1);
			queue.CopyBuffer(bufferR, bufferP, null);

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
				double rr = this.VectorDotVector(bufferR, bufferR);
				this.Matrix_x_Vector(bufferAp, bufferA, bufferColumnIndeces, bufferNonzeroCounts, bufferP);
				double alpha = rr / this.VectorDotVector(bufferP, bufferAp);
				this.VectorPlusVector(bufferX, bufferX, bufferP, alpha);
				this.VectorPlusVector(bufferR, bufferR, bufferAp, -alpha);

				// 収束したかどうかを取得
				converged = this.IsConverged(this.MaxAbsolute(bufferR));

				// 収束していなかったら
				if(!converged)
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
		/// ベクトルの値をすべて指定した値にして初期化する
		/// </summary>
		/// <param name="vector">初期化するベクトル</param>
		/// <param name="value">代入値</param>
		void InitializeVector(ComputeBuffer<double> vector, double value = 0)
		{
			// 各要素同士の足し算を実行
			//  # 初期化先のベクトル
			//  # 初期値
			setAllVector.SetMemoryArgument(0, vector);
			setAllVector.SetValueArgument(1, value);
			queue.Execute(setAllVector, null, new long[] { this.Count }, null, null);
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
			queue.Execute(addVectorVector, null, new long[] { this.Count }, null, null);
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
			multiplyVectorVector.SetMemoryArgument(0, bufferForDot);
			multiplyVectorVector.SetMemoryArgument(1, left);
			multiplyVectorVector.SetMemoryArgument(2, right);
			queue.Execute(multiplyVectorVector, null, new long[] { this.Count }, null, null);

			// 計算する配列の要素数
			int targetSize = this.Count;

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)Math.Ceiling((double)targetSize / 2 / localSize) * localSize;

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionSum.SetMemoryArgument(0, bufferForDot);
				reductionSum.SetValueArgument(1, targetSize);
				reductionSum.SetLocalArgument(2, sizeof(double) * localSize);
				queue.Execute(reductionSum, null, new long[] { globalSize }, new long[] { localSize }, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / localSize;
			}

			// 結果を読み込み
			queue.ReadFromBuffer(bufferForDot, ref answerForReduction, true, 0, 0, 1, null);

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
		void Matrix_x_Vector(ComputeBuffer<double> answer, ComputeBuffer<double> matrix, ComputeBuffer<int> columnIndeces, ComputeBuffer<int> nonzeroCounts, ComputeBuffer<double> vector)
		{
			// 行列とベクトルの積を実行
			//  # 解を格納するベクトル
			//  # 行列
			//  # ベクトル
			//  # 列番号
			//  # 非ゼロ要素数
			matrix_x_Vector.SetMemoryArgument(0, answer);
			matrix_x_Vector.SetMemoryArgument(1, matrix);
			matrix_x_Vector.SetMemoryArgument(2, vector);
			matrix_x_Vector.SetMemoryArgument(3, columnIndeces);
			matrix_x_Vector.SetMemoryArgument(4, nonzeroCounts);
			queue.Execute(matrix_x_Vector, null, new long[] { this.Count }, null, null);
		}

		/// <summary>
		/// ベクトルの中で最大絶対値を探す
		/// </summary>
		/// <param name="target">対象のベクトル</param>
		double MaxAbsolute(ComputeBuffer<double> target)
		{
			// 対象のデータを複製
			queue.CopyBuffer(target, bufferForMax, null);

			// 計算する配列の要素数
			int targetSize = this.Count;

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)Math.Ceiling((double)targetSize / 2 / localSize) * localSize;

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionMax.SetMemoryArgument(0, bufferForMax);
				reductionMax.SetValueArgument(1, targetSize);
				reductionMax.SetLocalArgument(2, sizeof(double) * localSize);
				queue.Execute(reductionMax, null, new long[] { globalSize }, new long[] { localSize }, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / localSize;
			}

			// 結果を読み込み
			queue.ReadFromBuffer(bufferForMax, ref answerForReduction, true, 0, 0, 1, null);

			// 結果を返す
			return answerForReduction[0];
		}
	}
}