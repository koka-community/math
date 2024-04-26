/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

kk_math_vector__blasvector kk_vector_blasvector(kk_vector_t v, kk_context_t* ctx);

kk_vector_t kk_blasvector_vector(kk_math_vector__blasvector bv, kk_context_t* ctx);

double kk_blasvector_unsafe_get(kk_math_vector__blasvector bv, kk_ssize_t index, kk_context_t* ctx);

kk_unit_t kk_blasvector_unsafe_set(kk_math_vector__blasvector bv, kk_ssize_t index, double value, kk_context_t* ctx);

kk_math_vector__blasvector kk_blasvector_copy(kk_math_vector__blasvector bv, kk_context_t* ctx);