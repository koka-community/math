


kk_math_vector__blasvector kk_vector_blasvector(kk_vector_t v, kk_context_t* ctx) {
    kk_ssize_t length;
    kk_box_t* vec_buf = kk_vector_buf_borrow(v, &length, ctx);

    double* buf = malloc( sizeof(double) * length);

    for (kk_ssize_t i = 0; i < length; i++) {
        buf[i] = kk_double_unbox(vec_buf[i], KK_OWNED, ctx);
    }

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)buf, ctx);

    return kk_math_vector__new_Blasvector(length, owned_buf, ctx);
}

kk_vector_t kk_blasvector_vector(kk_math_vector__blasvector bv, kk_context_t* ctx) {

    double* buf = (double*)kk_intptr_unbox(bv.internal.owned, KK_OWNED, ctx);
    kk_vector_t out_vec = kk_vector_alloc(bv.length, kk_box_null(), ctx);

    kk_ssize_t length;

    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &length, ctx);

    for (kk_ssize_t i = 0; i < bv.length; i++) {
        out_vec_buf[i] = kk_double_box(buf[i], ctx);
    }

    return out_vec;
}

double kk_blasvector_unsafe_get(kk_math_vector__blasvector bv, kk_ssize_t index, kk_context_t* ctx) {
    return ((double*)kk_intptr_unbox(bv.internal.owned, KK_OWNED, ctx))[index];
}

kk_unit_t kk_blasvector_unsafe_set(kk_math_vector__blasvector bv, kk_ssize_t index, double value, kk_context_t* ctx) {
    ((double*)kk_intptr_unbox(bv.internal.owned, KK_OWNED, ctx))[index] = value;
    return kk_Unit;
}

kk_math_vector__blasvector kk_blasvector_copy(kk_math_vector__blasvector bv, kk_context_t* ctx) {
    double* buf = malloc( sizeof(double) * bv.length);
    double* old_buf = (double*)kk_intptr_unbox(bv.internal.owned, KK_OWNED, ctx);

    for (kk_ssize_t i = 0; i < bv.length; i++) {
        buf[i] = old_buf[i];
    }

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)buf, ctx);

    return kk_math_vector__new_Blasvector(bv.length, owned_buf, ctx);
}