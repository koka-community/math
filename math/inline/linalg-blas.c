#include <cblas.h>


double kk_asum(kk_math_vector__blasvector bv, kk_context_t* ctx) {
    

    double result = cblas_dasum(bv.length, (double*)bv.internal, 1);

    kk_free((double*)bv.internal, ctx);
    bv.internal = 0;

    return result;
}

kk_math_vector__blasvector kk_axpy(kk_math_vector__blasvector a, kk_math_vector__blasvector b, double scalar, kk_context_t* ctx) {
    

    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;

    b.length = length;

    cblas_daxpy(length, scalar, (double*)a.internal, 1, (double*)b.internal, 1);


    kk_free((double*)a.internal, ctx);
    a.internal = 0;

    return b;
}

double kk_dot(kk_math_vector__blasvector a, kk_math_vector__blasvector b, kk_context_t* ctx) {
    
    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;

    double result = cblas_ddot(length, (double*)a.internal, 1, (double*)b.internal, 1);

    kk_free((double*)a.internal, ctx);
    kk_free((double*)b.internal, ctx);
    a.internal = 0;
    b.internal = 0;

    return result;
}

double kk_nrm2(kk_math_vector__blasvector bv, kk_context_t* ctx) {

    double result = cblas_dnrm2(bv.length, (double*)bv.internal, 1);

    kk_free((double*)bv.internal, ctx);
    bv.internal = 0;

    return result;
}

kk_std_core_types__tuple2 kk_rot(kk_math_vector__blasvector a, kk_math_vector__blasvector b, double scalar1, double scalar2, kk_context_t* ctx) {
    
    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;

    cblas_drot(length, (double*)a.internal, 1, (double*)b.internal, 1, scalar1, scalar2);

    kk_vector_t c_vec = kk_vector_alloc(length, kk_box_null(), ctx);
    kk_vector_t d_vec = kk_vector_alloc(length, kk_box_null(), ctx);

    kk_box_t* c_vec_buf = kk_vector_buf_borrow(c_vec, &length, ctx);
    kk_box_t* d_vec_buf = kk_vector_buf_borrow(d_vec, &length, ctx);

    kk_box_t a_box = kk_math_vector__blasvector_box(a, ctx);
    kk_box_t b_box = kk_math_vector__blasvector_box(b, ctx);

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

kk_std_core_types__tuple2 kk_rotm(kk_math_vector__blasvector a, kk_math_vector__blasvector b, double flag, kk_math_matrix__blasmatrix h_matrix, kk_context_t* ctx) {

    kk_ssize_t length = a.length;
    if (length > b.length) 
        length = b.length;

    
    double h[5] = { flag, ((double*)h_matrix.internal)[0], ((double*)h_matrix.internal)[1], ((double*)h_matrix.internal)[2], ((double*)h_matrix.internal)[3]};

    cblas_drotm(length, (double*)a.internal, 1, (double*)b.internal, 1, h);

    kk_free((double*)h_matrix.internal, ctx);
    h_matrix.internal = 0;

    kk_box_t a_box = kk_math_vector__blasvector_box(a, ctx);
    kk_box_t b_box = kk_math_vector__blasvector_box(b, ctx);


    return kk_std_core_types__new_Tuple2(a_box, b_box, ctx);
}

kk_std_core_types__tuple3 kk_rotmg(double d1, double d2, double x1, double y1, kk_context_t* ctx) {

    double param[5] = { 0 };

    cblas_drotmg(&d1, &d2, &x1, y1, param);

    double* h_matrix_buf = kk_malloc( sizeof(double) * 4, ctx);

    kk_math_matrix__blasmatrix h_matrix = kk_math_matrix__new_Blasmatrix(2, 2, (long int)h_matrix_buf, ctx);

    kk_box_t h_matrix_box = kk_math_matrix__blasmatrix_box(h_matrix, ctx);

    kk_std_core_types__tuple4 tuple = kk_std_core_types__new_Tuple4(kk_reuse_null, 0, kk_double_box(d1, ctx), kk_double_box(d2, ctx), kk_double_box(x1, ctx), kk_double_box(y1, ctx), ctx);
    kk_box_t tuple_box = kk_std_core_types__tuple4_box(tuple, ctx);

    return kk_std_core_types__new_Tuple3(tuple_box, kk_double_box(param[0], ctx), h_matrix_box, ctx);
}

kk_math_vector__blasvector kk_scal(kk_math_vector__blasvector bv, double scalar, kk_context_t* ctx) {

    cblas_dscal(bv.length, scalar, (double*)bv.internal, 1);
    
    return bv;
}

kk_std_core_types__tuple2 kk_iamax(kk_math_vector__blasvector bv, kk_context_t* ctx) {
    
    int64_t i = cblas_idamax(bv.length, (double*)bv.internal, 1);

    kk_box_t bv_box = kk_math_vector__blasvector_box(bv, ctx);
    kk_box_t int_box = kk_integer_box(kk_integer_from_uint64(i, ctx), ctx);

    return kk_std_core_types__new_Tuple2(int_box, bv_box, ctx);
}

kk_std_core_types__tuple2 kk_iamin(kk_math_vector__blasvector bv, kk_context_t* ctx) {

    int64_t i = cblas_idamin(bv.length, (double*)bv.internal, 1);

    kk_box_t bv_box = kk_math_vector__blasvector_box(bv, ctx);
    kk_box_t int_box = kk_integer_box(kk_integer_from_uint64(i, ctx), ctx);

    return kk_std_core_types__new_Tuple2(int_box, bv_box, ctx);
}









