/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// This a module that maps directly to BLAS calls
module math/linalg-raw
import std/test
import std/num/float64
import math/matrix

extern import 
  c file "inline/linalg-raw"

// Computes the sum of magnitudes of the vector elements.
pub extern asum( v : vector<float64> ) : float64
  c "kk_asum"

// Computes a vector-scalar product and adds the result to a vector.
pub extern axpy( x : vector<float64>, y : vector<float64>, scalar : float64 ) : vector<float64>
  c "kk_axpy"

// Computes the dot product of two real vectors.
pub extern dot( a : vector<float64>, b : vector<float64> ) : float64
  c "kk_dot"

// Computes the Euclidean norm of a vector.
pub extern nrm2( v : vector<float64> ) : float64
  c "kk_nrm2"

// Performs rotation of points in the plane.
pub extern rot( a : vector<float64>, b : vector<float64>, c : float64, s : float64 ) : (vector<float64>, vector<float64>)
  c "kk_rot"

// Computes the parameters for a Givens rotation.
// Given the Cartesian coordinates `(a, b)` of a point, the `rotg` routines return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation. 
pub extern rotg( a : float64, b: float64 ) : (float64, float64, float64, float64)
  c "kk_rotg"

// Performs modified Givens rotation of points in the plane.
// Returns (x, y)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotm( x : vector<float64>, y : vector<float64>, flag : float64, h : matrix<float64> ) : (vector<float64>, vector<float64>)
  c "kk_rotm"

// Computes the parameters for a modified Givens rotation.
// Returns ((d1, d2, x1, x2), flag, h)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotmg( d1 : float64, d2 : float64, x1 : float64, y1 : float64 ) : ((float64, float64, float64, float64), float64, matrix<float64>)
  c "kk_rotmg"

// Computes the product of a vector by a scalar.
pub extern scal( v : vector<float64>, scalar : float64 ) : vector<float64>
  c "kk_scal"

// Finds the index of the element with the largest absolute value in a vector.
pub extern iamax( x : vector<float64> ) : int
  c "kk_iamax"

// Finds the index of the element with the smallest absolute value.
pub extern iamin( x : vector<float64> ) : int
  c "kk_iamin"
