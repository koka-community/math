/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// This a module that maps directly to BLAS calls
module math/linalg-blas
import std/test
import std/num/float64
import math/matrix
import math/vector

/*
  Important Note:
  All of these functions will free up the blasvector or blasmatrix if it doesn't return it as an output.
  As a result it is best if you only continue using the blasvector or blasmatrix that has been returned from the function

*/


extern import 
  c file "inline/linalg-blas"

// Computes the sum of magnitudes of the vector elements.
pub extern asum( v : blasvector ) : float64
  c "kk_asum"

// Computes a vector-scalar product and adds the result to a vector.
pub extern axpy( x : blasvector, y : blasvector, scalar : float64 ) : blasvector
  c "kk_axpy"

// Computes the dot product of two real vectors.
pub extern dot( a : blasvector, b : blasvector ) : float64
  c "kk_dot"

// Computes the Euclidean norm of a vector.
pub extern nrm2( v : blasvector ) : float64
  c "kk_nrm2"

// Performs rotation of points in the plane.
pub extern rot( a : blasvector, b : blasvector, c : float64, s : float64 ) : (blasvector, blasvector)
  c "kk_rot"

// Computes the parameters for a Givens rotation.
// Given the Cartesian coordinates `(a, b)` of a point, the `rotg` routines return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation. 
pub extern rotg( a : float64, b: float64 ) : (float64, float64, float64, float64)
  c "kk_rotg"

// Performs modified Givens rotation of points in the plane.
// Returns (x, y)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotm( x : blasvector, y : blasvector, flag : float64, h : blasmatrix ) : (blasvector, blasvector)
  c "kk_rotm"

// Computes the parameters for a modified Givens rotation.
// Returns ((d1, d2, x1, x2), flag, h)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotmg( d1 : float64, d2 : float64, x1 : float64, y1 : float64 ) : ((float64, float64, float64, float64), float64, blasmatrix)
  c "kk_rotmg"

// Computes the product of a vector by a scalar.
pub extern scal( v : blasvector, scalar : float64 ) : blasvector
  c "kk_scal"

// Finds the index of the element with the largest absolute value in a vector.
pub extern iamax( x : blasvector ) : (int, blasvector)
  c "kk_iamax"

// Finds the index of the element with the smallest absolute value.
pub extern iamin( x : blasvector ) : (int, blasvector)
  c "kk_iamin"
