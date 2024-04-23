


kk_math_vector__blasvector kk_vector_blasvector(kk_vector_t v, kk_context_t* ctx) {
    kk_ssize_t length;
    kk_box_t* vec_buf = kk_vector_buf_borrow(v, &length, ctx);

    double* buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        buf[i] = kk_double_unbox(vec_buf[i], KK_OWNED, ctx);
    }

    return kk_math_vector__new_Blasvector(length, (long int)buf, ctx);
}

kk_vector_t kk_blasvector_vector(kk_math_vector__blasvector bv, kk_context_t* ctx) {

    double* buf = (double*)bv.internal;
    kk_vector_t out_vec = kk_vector_alloc(bv.length, kk_box_null(), ctx);

    kk_ssize_t length;

    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &length, ctx);

    for (kk_ssize_t i = 0; i < bv.length; i++) {
        out_vec_buf[i] = kk_double_box(buf[i], ctx);
    }
    kk_free(buf, ctx);

    return out_vec;
}