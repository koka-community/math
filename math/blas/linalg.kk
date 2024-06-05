/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// Bindings for OpenBLAS for Koka
// To compile make sure that you set the linker flags: -lm -lpthread gfortran -lopenblas
// Here is a snippet to help: `--cclinkopts=-lm --cclinkopts=-lpthread --cclinkopts=-lgfortran --cclinkopts=-lopenblas`
module math/blas/linalg
import std/test
import std/num/int32
import std/num/float64
import std/core-extras
import math/matrix
import math/blas/matrix
import math/blas/vector

pub import math/blas/raw

// Computes the sum of magnitudes of the vector elements.
// Level 1 BLAS
pub inline fun sum-of-magnitudes( bv : blasvector ) : float64
  asum( bv )

// Multiplies scalar with `bv1` and adds the result to `bv2`
// Level 1 BLAS
pub inline fun scalar-multiply-with-add( bv1 : blasvector, bv2 : blasvector, scalar : float64 ) : blasvector
  val bvo = bv1.axpy( bv2.copy, scalar )
  bvo

// Performs a dot product between two vectors of `float64`.
// Note: the smallest vector will be the maximum size for this operation
pub inline fun dot( a : blasvector, b : blasvector ) : float64
  a.raw/dot( b )

// Computes the Euclidean norm of a vector.
pub inline fun norm( bv : blasvector ) : float64
  bv.nrm2

// Performs rotation of points in the plane.
// Level 1 BLAS
pub inline fun rotate( a : blasvector, b : blasvector, c : float64, s : float64 ) : (blasvector, blasvector)
  rot( a.copy, b.copy, c, s )

// Computes the parameters for a Givens rotation.
// Level 1 BLAS
// Given the Cartesian coordinates `(a, b)` of a point, the `rotg` routines return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation. 
pub inline fun givens-rotation-parameters( a : float64, b : float64 ) : (float64, float64, float64, float64)
  rotg( a, b )

// Performs modified Givens rotation of points in the plane.
// Note: `h` is a 2x2 blasmatrix
// Level 1 BLAS
// Returns (x, y)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub inline fun modified-givens-rotation( a : blasvector, b : blasvector, flag : float64, h : blasmatrix ) : (blasvector, blasvector)
  rotm( a.copy, b.copy, flag, h)

// Computes the parameters for a modified Givens rotation.
// Level 1 BLAS
// Returns ((d1, d2, x1, x2), flag, h)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub inline fun modifed-givens-rotation-parameters( d1 : float64, d2 : float64, x1 : float64, y1 : float64 ) : ((float64, float64, float64, float64), float64, blasmatrix)
  rotmg( d1, d2, x1, y1 )

// Computes the product of a vector by a scalar.
// Level 1 BLAS
pub inline fun scalar-multiply( bv : blasvector, scalar : float64 ) : blasvector
  scal( bv.copy, scalar )


// Computes a matrix-vector product using a general matrix
// Level 2 BLAS
pub inline fun general/matrix-vector-product( m : blasmatrix, v : blasvector, alpha : float64 = 1.0, row-major : bool = True, transposed : bool = False ) : maybe<blasvector>
  if transposed && v.length != m.rows then
    return Nothing
  else if transposed.not && v.length != m.cols then
    return Nothing
  else
    Just( gemv( alpha, m, v.copy, 0.0, blasvector( v.length.int ), row-major, transpose = transposed ) )

// Computes a rank-1 update of a general matrix
// Level 2 BLAS
pub inline fun general/rank-one-update( m : blasmatrix, v1 : blasvector, v2 : blasvector, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasmatrix>
  if v1.length != m.rows || v2.length != m.cols then
    return Nothing
  else
    Just( ger( alpha, m.copy, v1, v2, row-major ) )

// Computes a matrix-vector product for a symmetric matrix
// Level 2 BLAS
pub inline fun symmetric/matrix-vector-product( m : blasmatrix, v : blasvector, upper-triangular : bool, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasvector>
  if v.length != m.rows then
    return Nothing
  else
    Just( symv( alpha, m, v.copy, 0.0, blasvector( v.length.int ), row-major, upper-triangular ) )

// Computes a rank-1 update of a symmetric matrix
// Level 2 BLAS
pub inline fun symmetric/rank-one-update( m : blasmatrix, v : blasvector, upper-triangular : bool, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasmatrix>
  if v.length != m.rows then
    return Nothing
  else
    Just( syr( alpha, m.copy, v, row-major, upper = upper-triangular ) )

// Computes a rank-2 update of a symmetric matrix
// Level 2 BLAS
pub inline fun symmetric/rank-two-update( m : blasmatrix, v1 : blasvector, v2 : blasvector, upper-triangular : bool, alpha : float64 = 1.0, row-major : bool = True ) : maybe<blasmatrix>
  if v1.length != m.rows || v2.length != m.rows then
    return Nothing
  else
    Just( syr2( alpha, m.copy, v1, v2, row-major, upper = upper-triangular ) )

// Computes a matrix-vector product using a triangular matrix.
// Level 2 BLAS
pub inline fun triangular/matrix-vector-product( m : blasmatrix, v : blasvector, upper-triangular : bool, row-major : bool = True, transposed : bool = False, unit-triangular : bool = False ) : maybe<blasvector>
  if v.length != m.rows then
    return Nothing
  else
    Just( trmv( m, v.copy, row-major, upper-triangular, transposed, unit-triangular ) )

// Solves a system of linear, equations whose coefficients are in a triangular matrix
// Level 2 BLAS
pub inline fun triangular/matrix-solver( m : blasmatrix, v : blasvector, upper-triangular : bool, row-major : bool = True, transposed : bool = False, unit-triangular : bool = False ) : maybe<blasvector>
  if v.length != m.rows then
    return Nothing
  else
    Just( trsv( m, v.copy, row-major, upper-triangular, transposed, unit-triangular ) )
