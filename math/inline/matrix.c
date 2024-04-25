#include <cblas.h>

kk_math_matrix__blasmatrix kk_matrix_blasmatrix(kk_vector_t matrix, kk_context_t* ctx) {
    kk_ssize_t len;
    kk_box_t* matrix_buf = kk_vector_buf_borrow(matrix, &len, ctx);

    kk_ssize_t index = 0;

    kk_ssize_t col_len;

    double* buf = NULL;
    for (kk_ssize_t i = 0; i < len; i++) {
        kk_box_t* col_buf = kk_vector_buf_borrow(kk_vector_unbox(matrix_buf[i],ctx), &col_len, ctx);
        if (buf == NULL) {
            printf("allocating\n");
            buf = malloc(sizeof(double) * len * col_len);
            printf("%p\n", buf);
        }
        for (kk_ssize_t j = 0; j < col_len; j++) {
            printf("indexing %ld\n", index);
            buf[index] = kk_double_unbox(col_buf[j], KK_OWNED, ctx);
            index += 1;
        }
    }

    printf("boxing c pointer\n");
    kk_box_t box = kk_std_cextern_c_own_extern(buf, ctx);
    printf("owning c pointer\n");
    kk_std_cextern__owned_c owned_buf = kk_std_cextern__new_Owned_c(box, ctx);
    printf("creating blasmatrix");
    return kk_math_matrix__new_Blasmatrix(len, col_len, owned_buf, ctx);
}

kk_vector_t kk_blasmatrix_matrix(kk_math_matrix__blasmatrix bmatrix, kk_context_t* ctx) {
    double* buf = (double*)kk_cptr_raw_unbox_borrowed(bmatrix.internal.owned, ctx);
    kk_vector_t out_vec = kk_vector_alloc(bmatrix.cols, kk_box_null(), ctx);

    kk_ssize_t length;
    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &length, ctx);

    kk_ssize_t index = 0;

    kk_ssize_t col_counter = 0;
    for (kk_ssize_t i = 0; i < (bmatrix.cols * bmatrix.rows); i += bmatrix.cols) {
        kk_ssize_t len;
        kk_vector_t col_vec = kk_vector_alloc(bmatrix.rows, kk_box_null(), ctx);
        kk_box_t* col_buf = kk_vector_buf_borrow(col_vec, &len, ctx);

        for (kk_ssize_t j = i; j < i + bmatrix.rows; j++) {
            col_buf[j - i] = kk_double_box(buf[j], ctx);
            index += 1;
        }

        out_vec_buf[col_counter] = kk_vector_box(col_vec, ctx);

        col_counter += 1;
    }

    return out_vec;
}


double kk_blasmatrix_unsafe_get(kk_math_matrix__blasmatrix bm, kk_ssize_t col, kk_ssize_t row, kk_context_t* ctx) {
    return ((double*)kk_intptr_unbox(bm.internal.owned, KK_OWNED, ctx))[col + row];
}

kk_unit_t kk_blasmatrix_unsafe_set(kk_math_matrix__blasmatrix bm, kk_ssize_t col, kk_ssize_t row, double value, kk_context_t* ctx) {
    ((double*)kk_intptr_unbox(bm.internal.owned, KK_OWNED, ctx))[col + row] = value;
    return kk_Unit;
}

kk_math_matrix__blasmatrix kk_blasmatrix_copy(kk_math_matrix__blasmatrix bm, kk_context_t* ctx) {
    double* buf = malloc( sizeof(double) * (bm.cols * bm.rows));
    double* old_buf = (double*)kk_cptr_raw_unbox_borrowed(bm.internal.owned, ctx);

    // This should be conditionally compiled if we ever get MAGMA bindings
    cblas_dcopy((bm.cols * bm.rows), old_buf, 1, buf, 1);
    /*for (kk_ssize_t i = 0; i < (bm.cols * bm.rows); i++) {
        buf[i] = old_buf[i];
    }*/

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)buf, ctx);

    return kk_math_matrix__new_Blasmatrix(bm.cols, bm.rows, owned_buf, ctx);
}