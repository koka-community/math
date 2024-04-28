/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// Bindings for OpenBLAS for Koka
// To compile make sure that you set the linker flags: -lm -lpthread gfortran -lopenblas
// Here is a snippet to help: `--cclinkopts=-lm --cclinkopts=-lpthread --cclinkopts=-lgfortran --cclinkopts=-lopenblas`
module math/linalg-blas
import std/test
import std/num/int32
import std/num/float64
import std/core-extras
import math/matrix
import math/vector
import math/linalg-blas-raw

// Computes the sum of magnitudes of the vector elements.
// Level 1 BLAS
pub inline fun blas/vector/sum-of-magnitudes( v : vector<float64> ) : float64
  val bv = v.blasvector
  blas/asum( bv )

// Computes the sum of magnitudes of the vector elements.
// Level 1 BLAS
pub inline fun blas/blasvector/sum-of-magnitudes( bv : blasvector ) : float64
  blas/asum( bv )

// Multiplies scalar with `v1` and adds the result to `v2`
// Level 1 BLAS
pub inline fun blas/vector/scalar-multiply-with-add( v1 : vector<float64>, v2 : vector<float64>, scalar : float64 ) : vector<float64>
  val bv1 = v1.blasvector
  val bv2 = v2.blasvector
  val bvo = bv1.blas/axpy( bv2, scalar )
  bvo.blasvector/vector

// Multiplies scalar with `bv1` and adds the result to `bv2`
// Level 1 BLAS
pub inline fun blas/blasvector/scalar-multiply-with-add( bv1 : blasvector, bv2 : blasvector, scalar : float64 ) : blasvector
  val bvo = bv1.blas/axpy( bv2.copy, scalar )
  bvo

// Performs a dot product between two vectors of `float64`.
// Note: the smallest vector will be the maximum size for this operation
pub inline fun blas/vector/dot( a : vector<float64>, b : vector<float64> ) : float64
  val x = a.blasvector
  val y = b.blasvector
  x.blas/dot( y )

// Performs a dot product between two vectors of `float64`.
// Note: the smallest vector will be the maximum size for this operation
pub inline fun blas/blasvector/dot( a : blasvector, b : blasvector ) : float64
  a.blas/dot( b )

// Performs the Euclidean norm of a vector.
pub inline fun blas/vector/norm( v : vector<float64> ) : float64
  val bv = v.blasvector
  bv.blas/nrm2

// Computes the Euclidean norm of a vector.
pub inline fun blas/blasvector/norm( bv : blasvector ) : float64
  bv.blas/nrm2

// Performs rotation of points in the plane.
// Level 1 BLAS 
pub inline fun blas/vector/rotate( a : vector<float64>, b : vector<float64>, c : float64, s : float64 ) : (vector<float64>, vector<float64>)
  val bv1 = a.blasvector
  val bv2 = b.blasvector
  val (out1, out2) = bv1.blas/rot( bv2, c, s )
  (out1.vector, out2.vector)

// Performs rotation of points in the plane.
// Level 1 BLAS
pub inline fun blas/blasvector/rotate( a : blasvector, b : blasvector, c : float64, s : float64 ) : (blasvector, blasvector)
  blas/rot( a.copy, b.copy, c, s )

// Computes the parameters for a Givens rotation.
// Level 1 BLAS
// Given the Cartesian coordinates `(a, b)` of a point, the `rotg` routines return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation. 
pub inline fun blas/givens-rotation-parameters( a : float64, b : float64 ) : (float64, float64, float64, float64)
  blas/rotg( a, b )

// Performs modified Givens rotation of points in the plane.
// Note: `h` is a 2x2 blasmatrix
// Level 1 BLAS
// Returns (x, y)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub inline fun blas/blasvector/modified-givens-rotation( a : blasvector, b : blasvector, flag : float64, h : blasmatrix ) : (blasvector, blasvector)
  blas/rotm( a.copy, b.copy, flag, h)

// Computes the parameters for a modified Givens rotation.
// Level 1 BLAS
// Returns ((d1, d2, x1, x2), flag, h)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub inline fun blas/blasvector/modifed-givens-rotation-parameters( d1 : float64, d2 : float64, x1 : float64, y1 : float64 ) : ((float64, float64, float64, float64), float64, blasmatrix)
  blas/rotmg( d1, d2, x1, y1 )

pub inline fun blas/vector/scalar-multiply( v : vector<float64>, scalar : float64 ) : vector<float64>
  val bv = v.blasvector
  blas/scal( bv, scalar ).vector

// Computes the product of a vector by a scalar.
// Level 1 BLAS
pub inline fun blas/blasvector/scalar-multiply( bv : blasvector, scalar : float64 ) : blasvector
  blas/scal( bv.copy, scalar )


// Computes a matrix-vector product using a general matrix
// Level 2 BLAS
pub inline fun blas/blasmatrix/general/matrix-vector-product( m : blasmatrix, v : blasvector, alpha : float64 = 1.0, row-major : bool = True, transposed : bool = False ) : maybe<blasvector>
  if transposed && v.length != m.rows then
    return Nothing
  else if transposed.not && v.length != m.cols then
    return Nothing
  else
    Just( blas/gemv( alpha, m, v.copy, 0.0, blasvector( v.length.int ), row-major, transpose = transposed ) )

// Computes a rank-1 update of a general matrix
// Level 2 BLAS
pub inline fun blas/blasmatrix/general/rank-one-update( m : blasmatrix, v1 : blasvector, v2 : blasvector, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasmatrix>
  if v1.length != m.rows || v2.length != m.cols then
    return Nothing
  else
    Just( blas/ger( alpha, m.copy, v1, v2, row-major ) )

// Computes a matrix-vector product for a symmetric matrix
// Level 2 BLAS
pub inline fun blas/blasmatrix/symmetric/matrix-vector-product( m : blasmatrix, v : blasvector, upper-triangular : bool, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasvector>
  if v.length != m.rows then
    return Nothing
  else
    Just( blas/symv( alpha, m, v.copy, 0.0, blasvector( v.length.int ), row-major, upper-triangular ) )

// Computes a rank-1 update of a symmetric matrix
// Level 2 BLAS
pub inline fun blas/blasmatrix/symmetric/rank-one-update( m : blasmatrix, v : blasvector, upper-triangular : bool, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasmatrix>
  if v.length != m.rows then
    return Nothing
  else
    Just( blas/syr( alpha, m.copy, v, row-major, upper = upper-triangular ) )

// Computes a rank-2 update of a symmetric matrix
// Level 2 BLAS
pub inline fun blas/blasmatrix/symmetric/rank-two-update( m : blasmatrix, v1 : blasvector, v2 : blasvector, upper-triangular : bool, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasmatrix>
  if v1.length != m.rows || v2.length != m.rows then
    return Nothing
  else
    Just( blas/syr2( alpha, m.copy, v1, v2, row-major, upper = upper-triangular ) )

// Computes a matrix-vector product using a triangular matrix.
// Level 2 BLAS
pub inline fun blas/blasmatrix/triangular/matrix-vector-product( m : blasmatrix, v : blasvector, upper-triangular : bool, row-major : bool = True, transposed : bool = False, unit-triangular : bool = False ) : maybe<blasvector>
  if v.length != m.rows then
    return Nothing
  else
    Just( blas/trmv( m, v.copy, row-major, upper-triangular, transposed, unit-triangular ) )

// Solves a system of linear, equations whose coefficients are in a triangular matrix
// Level 2 BLAS
pub inline fun blas/blasmatrix/triangular/matrix-solver( m : blasmatrix, v : blasvector, upper-triangular : bool, row-major : bool = True, transposed : bool = False, unit-triangular : bool = False ) : maybe<blasvector>
  if v.length != m.rows then
    return Nothing
  else
    Just( blas/trsv( m, v.copy, row-major, upper-triangular, transposed, unit-triangular ) )
