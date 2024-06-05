/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

#include <cblas.h>


double kk_asum(kk_math_blas_vector__blasvector bv, kk_context_t* ctx) {
    

    double result = cblas_dasum(bv.length, (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx), 1);

    return result;
}

kk_math_blas_vector__blasvector kk_axpy(kk_math_blas_vector__blasvector a, kk_math_blas_vector__blasvector b, double scalar, kk_context_t* ctx) {
    

    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;

    b.length = length;

    cblas_daxpy(length, scalar, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(b.internal.owned, ctx), 1);

    return b;
}

double kk_dot(kk_math_blas_vector__blasvector a, kk_math_blas_vector__blasvector b, kk_context_t* ctx) {
    
    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;
    
    double* a_buf = (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx);
    double* b_buf = (double*)kk_cptr_raw_unbox_borrowed(b.internal.owned, ctx);

    double result = cblas_ddot(length, a_buf, 1, b_buf, 1);

    return result;
}

double kk_nrm2(kk_math_blas_vector__blasvector bv, kk_context_t* ctx) {

    double result = cblas_dnrm2(bv.length, (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx), 1);

    return result;
}

kk_std_core_types__tuple2 kk_rot(kk_math_blas_vector__blasvector a, kk_math_blas_vector__blasvector b, double scalar1, double scalar2, kk_context_t* ctx) {
    
    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;

    cblas_drot(length, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(b.internal.owned, ctx), 1, scalar1, scalar2);

    kk_box_t a_box = kk_math_blas_vector__blasvector_box(a, ctx);
    kk_box_t b_box = kk_math_blas_vector__blasvector_box(b, ctx);

    return kk_std_core_types__new_Tuple2(a_box, b_box, ctx);
}

kk_std_core_types__tuple4 kk_rotg(double x, double y, kk_context_t* ctx) {
    double r = x;
    double z = y;
    double c = 0.0;
    double s = 0.0;

    cblas_drotg(&r, &z, &c, &s);

    return kk_std_core_types__new_Tuple4(kk_reuse_null, 0, kk_double_box(r, ctx), kk_double_box(z, ctx), kk_double_box(c, ctx), kk_double_box(s, ctx), ctx);
}

kk_std_core_types__tuple2 kk_rotm(kk_math_blas_vector__blasvector a, kk_math_blas_vector__blasvector b, double flag, kk_math_blas_matrix__blasmatrix h_matrix, kk_context_t* ctx) {

    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;
    
    double h[5] = { flag, ((double*)kk_cptr_raw_unbox_borrowed(h_matrix.internal.owned, ctx))[0], ((double*)kk_cptr_raw_unbox_borrowed(h_matrix.internal.owned, ctx))[2], ((double*)kk_cptr_raw_unbox_borrowed(h_matrix.internal.owned, ctx))[1], ((double*)kk_cptr_raw_unbox_borrowed(h_matrix.internal.owned, ctx))[3]};

    cblas_drotm(length, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(b.internal.owned, ctx), 1, h);

    kk_box_t a_box = kk_math_blas_vector__blasvector_box(a, ctx);
    kk_box_t b_box = kk_math_blas_vector__blasvector_box(b, ctx);


    return kk_std_core_types__new_Tuple2(a_box, b_box, ctx);
}

kk_std_core_types__tuple3 kk_rotmg(double d1, double d2, double x1, double y1, kk_context_t* ctx) {

    double param[5] = { 0 };

    cblas_drotmg(&d1, &d2, &x1, y1, param);

    double* h_matrix_buf = kk_malloc( sizeof(double) * 4, ctx);
    h_matrix_buf[0] = param[1];
    h_matrix_buf[1] = param[3];
    h_matrix_buf[2] = param[2];
    h_matrix_buf[3] = param[4];

    kk_std_cextern__owned_c owned_buf = kk_std_cextern_c_own((long int)h_matrix_buf, ctx);

    kk_math_blas_matrix__blasmatrix h_matrix = kk_math_blas_matrix__new_Blasmatrix(2, 2, owned_buf, ctx);

    kk_box_t h_matrix_box = kk_math_blas_matrix__blasmatrix_box(h_matrix, ctx);

    kk_std_core_types__tuple4 tuple = kk_std_core_types__new_Tuple4(kk_reuse_null, 0, kk_double_box(d1, ctx), kk_double_box(d2, ctx), kk_double_box(x1, ctx), kk_double_box(y1, ctx), ctx);
    kk_box_t tuple_box = kk_std_core_types__tuple4_box(tuple, ctx);

    return kk_std_core_types__new_Tuple3(tuple_box, kk_double_box(param[0], ctx), h_matrix_box, ctx);
}

kk_math_blas_vector__blasvector kk_scal(kk_math_blas_vector__blasvector bv, double scalar, kk_context_t* ctx) {

    cblas_dscal(bv.length, scalar, (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx), 1);
    
    return bv;
}

kk_integer_t kk_iamax(kk_math_blas_vector__blasvector bv, kk_context_t* ctx) {
    
    uint64_t i = cblas_idamax(bv.length, (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx), 1);

    return kk_integer_from_uint64(i, ctx);
}

kk_integer_t kk_iamin(kk_math_blas_vector__blasvector bv, kk_context_t* ctx) {

    uint64_t i = cblas_idamin(bv.length, (double*)kk_cptr_raw_unbox_borrowed(bv.internal.owned, ctx), 1);

    return kk_integer_from_uint64(i, ctx);
}

kk_math_blas_vector__blasvector kk_gemv(double scalar_a, kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, double scalar_b, kk_math_blas_vector__blasvector y, bool row_major, bool transpose, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }

    if (transpose) {
        cblas_dgemv(major, CblasTrans, a.rows, a.cols, scalar_a, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, scalar_b, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1);
    } else {
        cblas_dgemv(major, CblasNoTrans, a.rows, a.cols, scalar_a, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, scalar_b, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1);
    }
    return y;
}

kk_math_blas_matrix__blasmatrix kk_ger(double scalar, kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, kk_math_blas_vector__blasvector y, bool row_major, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }
    cblas_dger(major, a.rows, a.cols, scalar, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension);
    return a;
}

kk_math_blas_vector__blasvector kk_symv(double alpha, kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, double beta, kk_math_blas_vector__blasvector y, bool row_major, bool upper, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }

    if (upper) {
        cblas_dsymv(major, CblasUpper, a.rows, alpha, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, beta, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1);
    } else {
        cblas_dsymv(major, CblasLower, a.rows, alpha, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, beta, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1);
    }
    return y;
}

kk_math_blas_matrix__blasmatrix kk_syr(double alpha, kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, bool row_major, bool upper, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }
    
    if (upper) {
        cblas_dsyr(major, CblasUpper, a.rows, alpha, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension);
    } else {
        cblas_dsyr(major, CblasLower, a.rows, alpha, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension);
    }
    return a;
}

kk_math_blas_matrix__blasmatrix kk_syr2(double alpha, kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, kk_math_blas_vector__blasvector y, bool row_major, bool upper, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }

    if (upper) {
        cblas_dsyr2(major, CblasUpper, a.rows, alpha, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension);
    } else {
        cblas_dsyr2(major, CblasLower, a.rows, alpha, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(y.internal.owned, ctx), 1, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension);
    }
    return a;
}

kk_math_blas_vector__blasvector kk_trmv(kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, bool row_major, bool transposed, bool upper, bool unit_triangular, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }
    
    CBLAS_UPLO uplo;
    if (upper) {
        uplo = CblasUpper;
    } else {
        uplo = CblasLower;
    }
    CBLAS_TRANSPOSE transpose;
    if (transposed) {
        transpose = CblasTrans;
    } else {
        transpose = CblasNoTrans;
    }
    CBLAS_DIAG diag;
    if (unit_triangular) {
        diag = CblasUnit;
    } else {
        diag = CblasNonUnit;
    }

    cblas_dtrmv(major, uplo, transpose, diag, a.rows, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1);

    return x;
}

kk_math_blas_vector__blasvector kk_trsv(kk_math_blas_matrix__blasmatrix a, kk_math_blas_vector__blasvector x, bool row_major, bool transposed, bool upper_triangular, bool unit_triangular, kk_context_t* ctx) {
    CBLAS_ORDER major;
    int64_t leading_dimension;
    if (row_major) {
        major = CblasRowMajor;
        leading_dimension = a.rows;
    } else {
        major = CblasColMajor;
        leading_dimension = a.cols;
    }
    
    CBLAS_UPLO uplo;
    if (upper_triangular) {
        uplo = CblasUpper;
    } else {
        uplo = CblasLower;
    }
    CBLAS_TRANSPOSE transpose;
    if (transposed) {
        transpose = CblasTrans;
    } else {
        transpose = CblasNoTrans;
    }
    CBLAS_DIAG diag;
    if (unit_triangular) {
        diag = CblasUnit;
    } else {
        diag = CblasNonUnit;
    }

    cblas_dtrsv(major, uplo, transpose, diag, a.rows, (double*)kk_cptr_raw_unbox_borrowed(a.internal.owned, ctx), leading_dimension, (double*)kk_cptr_raw_unbox_borrowed(x.internal.owned, ctx), 1);

    return x;
}

