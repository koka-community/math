

double kk_blasmatrix_unsafe_get(kk_math_matrix__blasmatrix bm, kk_ssize_t col, kk_ssize_t row, kk_context_t* ctx) {
    return ((double*)kk_intptr_unbox(bm.internal.owned, KK_OWNED, ctx))[col + row];
}

kk_unit_t kk_blasmatrix_unsafe_set(kk_math_matrix__blasmatrix bm, kk_ssize_t col, kk_ssize_t row, double value, kk_context_t* ctx) {
    ((double*)kk_intptr_unbox(bm.internal.owned, KK_OWNED, ctx))[col + row] = value;
    return kk_Unit;
}

kk_math_matrix__blasmatrix kk_blasmatrix_copy(kk_math_matrix__blasmatrix bm, kk_context_t* ctx) {
    double* buf = malloc( sizeof(double) * (bm.cols * bm.rows));
    double* old_buf = (double*)kk_intptr_unbox(bm.internal.owned, KK_OWNED, ctx);

    for (kk_ssize_t i = 0; i < (bm.cols * bm.rows); i++) {
        buf[i] = old_buf[i];
    }

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)buf, ctx);

    return kk_math_matrix__new_Blasmatrix(bm.cols, bm.rows, owned_buf, ctx);
}