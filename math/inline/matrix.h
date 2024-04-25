/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

kk_math_matrix__blasmatrix kk_matrix_blasmatrix(kk_vector_t matrix, kk_context_t* ctx);

kk_vector_t kk_blasmatrix_matrix(kk_math_matrix__blasmatrix bmatrix, kk_context_t* ctx);

double kk_blasmatrix_unsafe_get(kk_math_matrix__blasmatrix bm, kk_ssize_t col, kk_ssize_t row, kk_context_t* ctx);

kk_unit_t kk_blasmatrix_unsafe_set(kk_math_matrix__blasmatrix bm, kk_ssize_t col, kk_ssize_t row, double value, kk_context_t* ctx);

kk_math_matrix__blasmatrix kk_blasmatrix_copy(kk_math_matrix__blasmatrix bm, kk_context_t* ctx);