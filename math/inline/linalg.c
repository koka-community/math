#include <cblas.h>


double kk_dot_product(kk_vector_t a, kk_vector_t b, kk_context_t* ctx) {
    kk_ssize_t a_length;
    kk_ssize_t b_length;
    kk_box_t* a_vec_buf = kk_vector_buf_borrow(a, &a_length, ctx);
    kk_box_t* b_vec_buf = kk_vector_buf_borrow(b, &b_length, ctx);

    kk_ssize_t length = a_length;
    if (length > b_length) 
        length = b_length;

    double* a_buf = kk_malloc( sizeof(double) * length, ctx);
    double* b_buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        a_buf[i] = kk_double_unbox(a_vec_buf[i], KK_OWNED, ctx);
        b_buf[i] = kk_double_unbox(b_vec_buf[i], KK_OWNED, ctx);
    }

    double result = cblas_ddot(length, a_buf, 1, b_buf, 1);

    kk_free(a_buf, ctx);
    kk_free(b_buf, ctx);

    return result;
}

double kk_vector_norm(kk_vector_t v, kk_context_t* ctx) {
    kk_ssize_t length;
    kk_box_t* vec_buf = kk_vector_buf_borrow(v, &length, ctx);

    double* buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        buf[i] = kk_double_unbox(vec_buf[i], KK_OWNED, ctx);
    }

    double result = cblas_dnrm2(length, buf, 1);

    kk_free(buf, ctx);

    return result;
}

// This should be a tuple but I am not sure how to create a tuple from C.
kk_vector_t kk_vector_rotation(kk_vector_t a, kk_vector_t b, double scalar1, double scalar2, kk_context_t* ctx) {
    kk_ssize_t a_length;
    kk_ssize_t b_length;
    kk_box_t* a_vec_buf = kk_vector_buf_borrow(a, &a_length, ctx);
    kk_box_t* b_vec_buf = kk_vector_buf_borrow(b, &b_length, ctx);

    kk_ssize_t length = a_length;
    if (length > b_length) 
        length = b_length;

    double* a_buf = kk_malloc( sizeof(double) * length, ctx);
    double* b_buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        a_buf[i] = kk_double_unbox(a_vec_buf[i], KK_OWNED, ctx);
        b_buf[i] = kk_double_unbox(b_vec_buf[i], KK_OWNED, ctx);
    }

    cblas_drot(length, a_buf, 1, b_buf, 1, scalar1, scalar2);

    kk_vector_t c_vec = kk_vector_alloc(length, kk_box_null(), ctx);
    kk_vector_t d_vec = kk_vector_alloc(length, kk_box_null(), ctx);

    kk_box_t* c_vec_buf = kk_vector_buf_borrow(c_vec, &length, ctx);
    kk_box_t* d_vec_buf = kk_vector_buf_borrow(d_vec, &length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        c_vec_buf[i] = kk_double_box(a_buf[i], ctx);
        d_vec_buf[i] = kk_double_box(b_buf[i], ctx);
    }

    kk_vector_t out_vec = kk_vector_alloc(2, kk_box_null(), ctx);
    kk_ssize_t out_vec_len;
    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &out_vec_len, ctx);
    out_vec_buf[0] = kk_vector_box(c_vec, ctx);
    out_vec_buf[1] = kk_vector_box(d_vec, ctx);

    kk_free(a_buf, ctx);
    kk_free(b_buf, ctx);

    return out_vec;
}

kk_vector_t kk_vector_rotation_givens_params(double x, double y, kk_context_t* ctx) {
    double r = x;
    double z = y;
    double c;
    double s;

    cblas_drotg(&r, &z, &c, &s);

    kk_vector_t out_vec = kk_vector_alloc(4, kk_box_null(), ctx);
    kk_ssize_t out_vec_len;
    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &out_vec_len, ctx);
    out_vec_buf[0] = kk_double_box(r, ctx);
    out_vec_buf[1] = kk_double_box(z, ctx);
    out_vec_buf[2] = kk_double_box(c, ctx);
    out_vec_buf[3] = kk_double_box(s, ctx);
}

kk_vector_t kk_modified_givens_rotation(kk_vector_t a, kk_vector_t b, kk_vector_t h_matrix, kk_context_t* ctx) {
    kk_ssize_t a_length;
    kk_ssize_t b_length;
    kk_box_t* a_vec_buf = kk_vector_buf_borrow(a, &a_length, ctx);
    kk_box_t* b_vec_buf = kk_vector_buf_borrow(b, &b_length, ctx);

    kk_ssize_t length = a_length;
    if (length > b_length) 
        length = b_length;

    double* a_buf = kk_malloc( sizeof(double) * length, ctx);
    double* b_buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        a_buf[i] = kk_double_unbox(a_vec_buf[i], KK_OWNED, ctx);
        b_buf[i] = kk_double_unbox(b_vec_buf[i], KK_OWNED, ctx);
    }

    kk_ssize_t h_length;
    kk_box_t* h_buf = kk_vector_buf_borrow(h_matrix, &h_length, ctx);
    kk_ssize_t h_length1;
    kk_box_t* h_buf1 = kk_vector_buf_borrow(kk_vector_unbox(h_buf[0], ctx), &h_length1, ctx);
    kk_ssize_t h_length2;
    kk_box_t* h_buf2 = kk_vector_buf_borrow(kk_vector_unbox(h_buf[1], ctx), &h_length2, ctx);

    double h[5] = { -1.0, kk_double_unbox( h_buf1[0], KK_OWNED, ctx), kk_double_unbox( h_buf2[0], KK_OWNED, ctx), kk_double_unbox( h_buf1[1], KK_OWNED, ctx), kk_double_unbox( h_buf2[1], KK_OWNED, ctx) };

    cblas_drotm(length, a_buf, 1, b_buf, 1, h);

    kk_vector_t c_vec = kk_vector_alloc(length, kk_box_null(), ctx);
    kk_vector_t d_vec = kk_vector_alloc(length, kk_box_null(), ctx);

    kk_box_t* c_vec_buf = kk_vector_buf_borrow(c_vec, &length, ctx);
    kk_box_t* d_vec_buf = kk_vector_buf_borrow(d_vec, &length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        c_vec_buf[i] = kk_double_box(a_buf[i], ctx);
        d_vec_buf[i] = kk_double_box(b_buf[i], ctx);
    }


    kk_free(a_buf, ctx);
    kk_free(b_buf, ctx);

    kk_vector_t out_vec = kk_vector_alloc(2, kk_box_null(), ctx);
    kk_ssize_t out_vec_len;
    kk_box_t* out_vec_buf = kk_vector_buf_borrow(out_vec, &out_vec_len, ctx);
    out_vec_buf[0] = kk_vector_box(c_vec, ctx);
    out_vec_buf[1] = kk_vector_box(d_vec, ctx);

    return out_vec;
}


kk_vector_t kk_scale_vector(kk_vector_t v, double scalar, kk_context_t* ctx) {
    kk_ssize_t length;
    kk_box_t* vec_buf = kk_vector_buf_borrow(v, &length, ctx);

    double* buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        buf[i] = kk_double_unbox(vec_buf[i], KK_OWNED, ctx);
    }

    cblas_dscal(length, scalar, buf, 1);
    kk_vector_t output;
    if (kk_datatype_is_unique(v, ctx)) {   
        output = v;
    } else {
        output = kk_vector_copy(v, ctx);
    }
    kk_box_t* out_buf = kk_vector_buf_borrow(output, &length, ctx);
    for (kk_ssize_t i = 0; i < length; i++) {
        out_buf[i] = kk_double_box(buf[i], ctx);
    } 
    kk_free(buf, ctx);

    return output;
}

double kk_sum_magnitudes(kk_vector_t v, kk_context_t* ctx) {
    kk_ssize_t length;
    kk_box_t* vec_buf = kk_vector_buf_borrow(v, &length, ctx);

    double* buf = kk_malloc( sizeof(double) * length, ctx);

    for (kk_ssize_t i = 0; i < length; i++) {
        buf[i] = kk_double_unbox(vec_buf[i], KK_OWNED, ctx);
    }

    double result = cblas_dasum(length, buf, 1);

    kk_free(buf, ctx);

    return result;
}
