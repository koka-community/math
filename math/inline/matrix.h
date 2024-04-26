/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

struct kk_math_matrix_Blasmatrix kk_matrix_blasmatrix(kk_vector_t matrix, kk_context_t* ctx);

kk_vector_t kk_blasmatrix_matrix(struct kk_math_matrix_Blasmatrix bmatrix, kk_context_t* ctx);

double kk_blasmatrix_unsafe_get(struct kk_math_matrix_Blasmatrix bm, kk_ssize_t col, kk_ssize_t row, kk_context_t* ctx);

kk_unit_t kk_blasmatrix_unsafe_set(struct kk_math_matrix_Blasmatrix bm, kk_ssize_t col, kk_ssize_t row, double value, kk_context_t* ctx);

struct kk_math_matrix_Blasmatrix kk_blasmatrix_copy(struct kk_math_matrix_Blasmatrix bm, kk_context_t* ctx);