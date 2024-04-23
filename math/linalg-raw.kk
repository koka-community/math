// This a module that maps directly to BLAS calls
module math/linalg-raw
import std/test
import std/num/float64
import math/matrix

extern import 
  c file "inline/linalg-raw"

// Return the element at position `index` in vector `v` without bounds check!
// This function should never be called directly unless you know exactly what you are doing
inline extern unsafe-idx( ^v : vector<a>, index : ssize_t ) : total a
  c  "kk_vector_at_borrow"
  cs inline "(#1)[#2]"
  js inline "(#1)[#2]"

pub extern asum( v : vector<float64> ) : float64
  c "kk_asum"

pub extern axpy( x : vector<float64>, y : vector<float64>, scalar : float64 ) : vector<float64>
  c "kk_axpy"

pub extern dot( a : vector<float64>, b : vector<float64> ) : float64
  c "kk_dot"

pub extern scal( v : vector<float64>, scalar : float64 ) : vector<float64>
  c "kk_scal"

pub extern nrm2( v : vector<float64> ) : float64
  c "kk_nrm2"

pub extern rot( a : vector<float64>, b : vector<float64>, c : float64, s : float64 ) : (vector<float64>, vector<float64>)
  c "kk_rot"

// Computes the parameters for a Givens rotation.
// Given the Cartesian coordinates `(a, b)` of a point, the `rotg` routines return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation. 
pub extern rotg( a : float64, b: float64 ) : (float64, float64, float64, float64)
  c "kk_rotg"

// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotm( x : vector<float64>, y : vector<float64>, flag : int64, h : matrix<float64> ) : (vector<float64>, vector<float64>)
  c "kk_rotm"

// Returns ((d1, d2, x1, x2), flag, h)
// Flag options are: -1.0, 0.0, 1.0, -2.0
pub extern rotmg( d1 : float64, d2 : float64, x1 : float64, y1 : float64 ) : ((float64, float64, float64, float64), float64, matrix<float64>)
  c "kk_rotmg"

pub extern iamax( x : vector<float64> ) : int
  c "kk_iamax"

pub extern iamin( x : vector<float64> ) : int
  c "kk_iamin"
