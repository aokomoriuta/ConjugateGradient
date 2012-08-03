//! ベクトルの和を計算する
/*!
	\param answer 解を設定するベクトル
	\param left 足されるベクトル
	\param right 足すベクトル
	\param C 足すベクトルにかける係数
*/
__kernel void Vector_dot_Vector(
	double double* answer,
	const double* left,
	const double* right,
	const double C)
{
	// 要素番号を取得
	int i = get_global_id(0);

	// 各々の要素を足して設定
	answer[i] = left[i] + C*right[i];
}
