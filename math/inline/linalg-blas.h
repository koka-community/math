/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

double kk_asum(kk_math_vector__blasvector bv, kk_context_t* ctx);

kk_math_vector__blasvector kk_axpy(kk_math_vector__blasvector a, kk_math_vector__blasvector b, double scalar, kk_context_t* ctx);

double kk_dot(kk_math_vector__blasvector a, kk_math_vector__blasvector b, kk_context_t* ctx);

double kk_nrm2(kk_math_vector__blasvector bv, kk_context_t* ctx);

kk_std_core_types__tuple2 kk_rot(kk_math_vector__blasvector a, kk_math_vector__blasvector b, double scalar1, double scalar2, kk_context_t* ctx);

kk_std_core_types__tuple4 kk_rotg(double x, double y, kk_context_t* ctx);

kk_std_core_types__tuple2 kk_rotm(kk_math_vector__blasvector a, kk_math_vector__blasvector b, double flag, kk_math_matrix__blasmatrix h_matrix, kk_context_t* ctx);

kk_std_core_types__tuple3 kk_rotmg(double d1, double d2, double x1, double y1, kk_context_t* ctx);

kk_math_vector__blasvector kk_scal(kk_math_vector__blasvector bv, double scalar, kk_context_t* ctx);

kk_integer_t kk_iamax(kk_math_vector__blasvector bv, kk_context_t* ctx);

kk_integer_t kk_iamin(kk_math_vector__blasvector bv, kk_context_t* ctx);









