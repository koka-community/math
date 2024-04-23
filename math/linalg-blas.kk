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



extern import 
  c file "inline/linalg-blas"

// Computes the sum of magnitudes of the vector elements.
// Level 1 BLAS
pub extern asum( v : blasvector ) : float64
  c "kk_asum"

// Computes a vector-scalar product and adds the result to a vector.
// Level 1 BLAS
// WARNING: This will mutate the `y` `blasvector`.
pub extern axpy( x : blasvector, y : blasvector, scalar : float64 ) : blasvector
  c "kk_axpy"

// Computes the dot product of two real vectors.
// Level 1 BLAS
pub extern dot( a : blasvector, b : blasvector ) : float64
  c "kk_dot"

// Computes the Euclidean norm of a vector.
// Level 1 BLAS
pub extern nrm2( v : blasvector ) : float64
  c "kk_nrm2"

// Performs rotation of points in the plane.
// Level 1 BLAS
// WARNING: This will mutate the `a` and `b` `blasvector`s.
pub extern rot( a : blasvector, b : blasvector, c : float64, s : float64 ) : (blasvector, blasvector)
  c "kk_rot"

// Computes the parameters for a Givens rotation.
// Level 1 BLAS
// Given the Cartesian coordinates `(a, b)` of a point, the `rotg` routines return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation. 
pub extern rotg( a : float64, b: float64 ) : (float64, float64, float64, float64)
  c "kk_rotg"

// Performs modified Givens rotation of points in the plane.
// Level 1 BLAS
// Returns (x, y)
// Flag options are: -1.0, 0.0, 1.0, -2.0
// WARNING: This will mutate the `x` and `y` `blasvector`s.
pub extern rotm( x : blasvector, y : blasvector, flag : float64, h : blasmatrix ) : (blasvector, blasvector)
  c "kk_rotm"

// Computes the parameters for a modified Givens rotation.
// Level 1 BLAS
// Returns ((d1, d2, x1, x2), flag, h)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotmg( d1 : float64, d2 : float64, x1 : float64, y1 : float64 ) : ((float64, float64, float64, float64), float64, blasmatrix)
  c "kk_rotmg"

// Computes the product of a vector by a scalar.
// Level 1 BLAS
// WARNING: This will mutate the `bv` `blasvector`
pub extern scal( bv : blasvector, scalar : float64 ) : blasvector
  c "kk_scal"

// Finds the index of the element with the largest absolute value in a vector.
// Level 1 BLAS
pub extern iamax( x : blasvector ) : int
  c "kk_iamax"

// Finds the index of the element with the smallest absolute value.
// Level 1 BLAS
pub extern iamin( x : blasvector ) : int
  c "kk_iamin"
