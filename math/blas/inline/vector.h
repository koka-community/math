/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

#include <cblas.h>


kk_math_blas_vector__blasvector kk_vector_blasvector(kk_vector_t v, kk_context_t* ctx) {
    kk_ssize_t length;
    kk_box_t* vec_buf = kk_vector_buf_borrow(v, &length, ctx);

    double* buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        buf[i] = kk_double_unbox(vec_buf[i], KK_OWNED, ctx);
    }

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)buf, ctx);

    return kk_math_blas_vector__new_Blasvector(length, owned_buf, ctx);
}

kk_vector_t kk_blasvector_vector(kk_math_blas_vector__blasvector bv, kk_context_t* ctx) {

    double* buf = (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx);
    kk_vector_t out_vec = kk_vector_alloc(bv.length, kk_box_null(), ctx);

    kk_ssize_t length;
    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &length, ctx);

    for (kk_ssize_t i = 0; i < bv.length; i++) {
        out_vec_buf[i] = kk_double_box(buf[i], ctx);
    }

    return out_vec;
}

double kk_blasvector_unsafe_get(kk_math_blas_vector__blasvector bv, kk_ssize_t index, kk_context_t* ctx) {
    return ((double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx))[index];
}

kk_unit_t kk_blasvector_unsafe_set(kk_math_blas_vector__blasvector bv, kk_ssize_t index, double value, kk_context_t* ctx) {
    ((double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx))[index] = value;
    return kk_Unit;
}

kk_math_blas_vector__blasvector kk_blasvector_copy(kk_math_blas_vector__blasvector bv, kk_context_t* ctx) {
    double* buf = kk_malloc( sizeof(double) * bv.length, ctx);
    double* old_buf = (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx);

    // This should be conditionally compiled if we ever get MAGMA bindings
    cblas_dcopy(bv.length, old_buf, 1, buf, 1);
    /*for (kk_ssize_t i = 0; i < bv.length; i++) {
        buf[i] = old_buf[i];
    }*/

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)buf, ctx);

    return kk_math_blas_vector__new_Blasvector(bv.length, owned_buf, ctx);
}