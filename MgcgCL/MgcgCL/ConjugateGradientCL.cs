using LWisteria.MgcgCL.LongVector;
using Cloo;
using System;

namespace LWisteria.MgcgCL
{
	/// <summary>
	/// 共役勾配法
	/// </summary>
	public class ConjugateGradientCL : ConjugateGradient
	{
		/// <summary>
		/// 内積で使うワークグループ内ワークアイテム数
		/// </summary>
		readonly long localSizeForDot;

		/// <summary>
		/// 行列とベクトルの積で使うワークグループ内ワークアイテム数
		/// </summary>
		readonly long localSizeForMatrix_x_Vector;

		/// <summary>
		/// コマンドキュー
		/// </summary>
		ComputeCommandQueue queue;

		#region カーネル
		/// <summary>
		/// ベクトルの要素を設定する
		/// </summary>
		ComputeKernel setAllVectorKernel;

		/// <summary>
		/// ベクトルの各要素を加算する
		/// </summary>
		ComputeKernel plusEachVectorKernel;

		/// <summary>
		/// ベクトルとベクトルの各要素の積算をする
		/// </summary>
		ComputeKernel multiplyEachVectorKernel;

		/// <summary>
		/// 行列とベクトルの各要素の積算をする
		/// </summary>
		ComputeKernel multiplyMatrixVectorKernel;

		/// <summary>
		/// 縦ベクトルを横ベクトルにする
		/// </summary>
		ComputeKernel columnVectorToRowKernel;

		/// <summary>
		/// ローカルメモリ内の総和を先頭に格納する
		/// </summary>
		ComputeKernel addEachLocalValuesToTopKernel;

		/// <summary>
		/// ローカルメモリ内の最大値を先頭に格納する
		/// </summary>
		ComputeKernel storeMaxEachLocalValuesToTopKernel;
		#endregion

		#region バッファー
		/// <summary>
		/// 係数行列
		/// </summary>
		ComputeBuffer<double> bufferA;

		/// <summary>
		/// 列番号
		/// </summary>
		ComputeBuffer<long> bufferAColumnIndeces;
		
		/// <summary>
		/// 非ゼロ要素数
		/// </summary>
		ComputeBuffer<long> bufferANonzeroCounts;

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
		/// 内積の解
		/// </summary>
		double[] answerForDot;

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
		/// 共役勾配法を生成する
		/// </summary>
		/// <param name="count">要素数</param>
		/// <param name="maxNonZeroCount"></param>
		/// <param name="_minIteration"></param>
		/// <param name="_maxIteration"></param>
		/// <param name="_allowableResidual"></param>
		public ConjugateGradientCL(long count, long maxNonZeroCount, int _minIteration, int _maxIteration, double allowableResidual)
			: base(count, maxNonZeroCount, _minIteration, _maxIteration, allowableResidual)
		{
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
			bufferForMax = new ComputeBuffer<double>(context, ComputeMemoryFlags.ReadWrite, this.Count);

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
			multiplyMatrixVectorKernel = program.CreateKernel("MultiplyMatrixVector");
			columnVectorToRowKernel = program.CreateKernel("ColumnVectorToRow");
			addEachLocalValuesToTopKernel = program.CreateKernel("AddEachLocalValuesToTop");
			storeMaxEachLocalValuesToTopKernel = program.CreateKernel("StoreMaxEachLocalValuesToTop");
			
			// 行列とベクトルの積の場合は、1行あたりの最大列数に近い2^nで1グループを回す
			this.localSizeForMatrix_x_Vector = Math.Min(
				(long)Math.Pow(2, Math.Ceiling(Math.Log(this.A.MaxNonzeroCountPerRow, 2))),
				devices[0].MaxWorkGroupSize);

			// 内積の計算の場合は、回せる最大の数
			this.localSizeForDot = devices[0].MaxWorkGroupSize;
		}


		/// <summary>
		/// OpenCLを使って方程式を解く
		/// </summary>
		override public void Solve()
		{
			// データを転送
			queue.WriteToBuffer(this.A.Elements, bufferA, false, null);
			queue.WriteToBuffer(this.b, bufferB, false, null);
			queue.WriteToBuffer(this.x, bufferX, false, null);
			queue.WriteToBuffer(this.A.ColumnIndeces, bufferAColumnIndeces, false, null);
			queue.WriteToBuffer(this.A.NonzeroCounts, bufferANonzeroCounts, false, null);

			this.Matrix_x_Vector(bufferX, bufferA, bufferAColumnIndeces, bufferANonzeroCounts, bufferB);

			//// 初期値を設定
			///*
			// * (Ap)_0 = A * x
			// * r_0 = b - Ap
			// * p_0 = (LDLr)_0
			// */
			//this.Matrix_x_Vector(bufferAp, bufferA, bufferAColumnIndeces, bufferANonzeroCounts, bufferX);
			//this.VectorPlusVector(bufferR, bufferB, bufferAp, -1);
			//queue.CopyBuffer(bufferR, bufferP, null);
			
			//// 収束したかどうか
			//bool converged = false;

			//// 収束しない間繰り返す
			//for(this.Iteration = 0; !converged; this.Iteration++)
			//{
			//    // 計算を実行
			//    /*
			//     * rr = r・r
			//     * Ap = A * p
			//     * α = rr/(p・Ap)
			//     * x' += αp
			//     * r' -= αAp
			//     */
			//    double rr = this.VectorDotVector(bufferR, bufferR);
			//    this.Matrix_x_Vector(bufferAp, bufferA, bufferAColumnIndeces, bufferANonzeroCounts, bufferP);

			//    var debug = new double[Count];
			//    queue.ReadFromBuffer(bufferAp, ref debug, true, null);

			//    Console.WriteLine("{0}: {1}, {2}", this.Iteration, rr, debug.Dot(debug));

			//    double alpha = rr / this.VectorDotVector(bufferP, bufferAp);
			//    this.VectorPlusVector(bufferX, bufferX, bufferP, alpha);
			//    this.VectorPlusVector(bufferR, bufferR, bufferAp, -alpha);

			//    // 収束したかどうかを取得
			//    converged = this.IsConverged(this.Max(bufferR));

			//    // 収束していなかったら
			//    if(!converged)
			//    {
			//        // 残りの計算を実行
			//        /*
			//         * β= r'r'/rLDLr
			//         * p = r' + βp
			//         * r'r' = r'・r'
			//         */
			//        double rrNew = this.VectorDotVector(bufferR, bufferR);
			//        double beta = rrNew / rr;
			//        this.VectorPlusVector(bufferP, bufferR, bufferP, beta);
			//    }
			//}

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
			setAllVectorKernel.SetMemoryArgument(0, vector);
			setAllVectorKernel.SetValueArgument(1, value);
			queue.Execute(setAllVectorKernel, null, new long[] { this.Count }, null, null);
		}


		/// <summary>
		/// ベクトル同士の和を計算する
		/// </summary>
		/// <param name="answer">解の代入先</param>
		/// <param name="left">足されるベクトル</param>
		/// <param name="right">足すベクトル</param>
		/// <param name="C">足すベクトルに掛ける係数</param>
		void VectorPlusVector(ComputeBuffer<double> answer, ComputeBuffer<double> left, ComputeBuffer<double> right, double C = 1.0)
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
			multiplyEachVectorKernel.SetMemoryArgument(0, bufferForDot);
			multiplyEachVectorKernel.SetMemoryArgument(1, left);
			multiplyEachVectorKernel.SetMemoryArgument(2, right);
			queue.Execute(multiplyEachVectorKernel, null, new long[] { this.Count }, null, null);

			var debug = new double[this.Count];
			queue.ReadFromBuffer(bufferForDot, ref debug, true, null);

			// リダクションで和をとる
			SumEachRow(1, this.Count, bufferForDot, this.localSizeForDot);

			// 結果を取得
			queue.ReadFromBuffer(bufferForDot, ref answerForDot, true, 0, 0, 1, null);

			// 結果を返す
			return answerForDot[0];
		}

		/// <summary>
		/// 行列とベクトルの積を計算する
		/// </summary>
		/// <param name="answer">解の代入先</param>
		/// <param name="matrix">行列</param>
		/// <param name="columnIndeces">列番号</param>
		/// <param name="nonzeroCounts">非ゼロ要素数</param>
		/// <param name="vector">ベクトル</param>
		void Matrix_x_Vector(ComputeBuffer<double> answer, ComputeBuffer<double> matrix, ComputeBuffer<long> columnIndeces, ComputeBuffer<long> nonzeroCounts, ComputeBuffer<double> vector)
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

			// リダクションで各行の和をとる
			SumEachRow(this.Count, this.A.MaxNonzeroCountPerRow, bufferForMatrix_x_Vector, localSizeForMatrix_x_Vector);

			// 縦ベクトルを横ベクトルに変換して、結果に格納
			//  # 配列
			//  # 行列
			//  # 行列の列数
			columnVectorToRowKernel.SetMemoryArgument(0, answer);
			columnVectorToRowKernel.SetMemoryArgument(1, bufferForMatrix_x_Vector);
			columnVectorToRowKernel.SetValueArgument(2, this.A.MaxNonzeroCountPerRow);
			queue.Execute(columnVectorToRowKernel, null, new long[] { this.Count }, null, null);
		}

		/// <summary>
		/// 行列の各行で総和を計算する
		/// </summary>
		/// <param name="rowCount">行数</param>
		/// <param name="columnCount">列数</param>
		/// <param name="target">総和をとる対象の行列</param>
		/// <param name="localSize">ワークグループ内ワークアイテム数</param>
		void SumEachRow(long rowCount, long columnCount, ComputeBuffer<double> target, long localSize)
		{
			// 以前の大きさを設定
			long oldSize = columnCount;

			// リダクションの計算が終了するまで書く大きさで
			for (long thisSize = oldSize; oldSize > 1; thisSize /= localSize)
			{
				// 前の大きさが奇数だった場合は1つ上の偶数にする
				thisSize = localSize * (long)Math.Ceiling((double)oldSize / localSize);

				// 後半の値を前半の値に加える（リダクション）
				//  # 今回計算する要素数
				//  # 1行あたりの要素数
				//  # 総和を実行する対象
				addEachLocalValuesToTopKernel.SetValueArgument(0, oldSize);
				addEachLocalValuesToTopKernel.SetValueArgument(1, columnCount);
				addEachLocalValuesToTopKernel.SetMemoryArgument(2, target);
				addEachLocalValuesToTopKernel.SetLocalArgument(3, sizeof(double) * localSize);
				queue.Execute(addEachLocalValuesToTopKernel, null, new long[] { rowCount, thisSize / 2 }, new long[] { 1L, localSize / 2 }, null);

				// 今回の大きさを保存
				oldSize = thisSize / localSize;
			}
		}

		/// <summary>
		/// 行列の各行で総和を計算する
		/// </summary>
		/// <param name="target">総和をとる対象の行列</param>
		double Max(ComputeBuffer<double> target)
		{
			queue.CopyBuffer(target, bufferForMax, null);

			// 以前の大きさを設定
			long oldSize = this.Count;

			long localSize = this.localSizeForDot;

			// リダクションの計算が終了するまで書く大きさで
			for(long thisSize = oldSize; oldSize > 1; thisSize /= localSize)
			{
				// 前の大きさが奇数だった場合は1つ上の偶数にする
				thisSize = localSize * (long)Math.Ceiling((double)oldSize / localSize);

				// 後半の値を前半の値に加える（リダクション）
				//  # 今回計算する要素数
				//  # 1行あたりの要素数
				//  # 総和を実行する対象
				storeMaxEachLocalValuesToTopKernel.SetValueArgument(0, oldSize);
				storeMaxEachLocalValuesToTopKernel.SetValueArgument(1, this.Count);
				storeMaxEachLocalValuesToTopKernel.SetMemoryArgument(2, bufferForMax);
				storeMaxEachLocalValuesToTopKernel.SetLocalArgument(3, sizeof(double) * localSize);
				queue.Execute(storeMaxEachLocalValuesToTopKernel, null, new long[] { 1L, thisSize / 2 }, new long[] { 1L, localSize / 2 }, null);

				// 今回の大きさを保存
				oldSize = thisSize / localSize;
			}

			var debug = new double[this.Count];
			queue.ReadFromBuffer(target, ref debug, true, null);

			answerForDot[0] = 0;

			// 結果を取得
			queue.ReadFromBuffer(bufferForMax, ref answerForDot, true, 0, 0, 1, null);

			// 結果を返す
			return answerForDot[0];
		}
	}
}