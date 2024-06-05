/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// Here is a module for a matrix data structure that is a wrapper around a C double pointer.
module math/blas/matrix
import std/num/float64
import std/num/int32
import std/cextern
import std/test
import math/matrix

extern import
  c file "inline/matrix.h"

// A wrapper around a C `double` pointer.
// A blasmatrix is neither column nor row major
pub value struct blasmatrix
  pub rows : int32
  pub cols : int32
  internal : owned-c<float64>

// Creates a `blasmatrix` from `cols` and `rows`.
pub fun blasmatrix( rows : int, cols : int ) : blasmatrix
  Blasmatrix( rows.int32, cols.int32, c-own( allocate-buffer( cols.int32, rows.int32 ) ) )

extern allocate-buffer( cols : int32, rows : int32 ) : intptr_t
  c inline "(long int)kk_malloc(sizeof(double) * #1 * #2, kk_context())"

// Creates a blasmatrix from a matrix<float> `m`.
// This will create a row major blasmatrix.
pub extern matrix/blasmatrix( m : matrix<float64> ) : blasmatrix
  c "kk_matrix_blasmatrix"

// Creates a matrix<float64> from a blasmatrix `bm`
pub extern blasmatrix/matrix( bm : blasmatrix, row-major : bool = True ) : matrix<float64>
  c "kk_blasmatrix_matrix"

// Get a value from a blasmatrix without a bounds check
inline extern unsafe-get( bm : blasmatrix, row : ssize_t, col : ssize_t ) : float64
  c "kk_blasmatrix_unsafe_get"

// Set a value in the blasmatrix without a bounds check
inline extern unsafe-set( bm : blasmatrix, row : ssize_t, col : ssize_t, value : float64 ) : ()
  c "kk_blasmatrix_unsafe_set"

// Copy a blasmatrix
// This uses the BLAS copy routine to make it fast.
pub extern copy( bv : blasmatrix ) : blasmatrix
  c "kk_blasmatrix_copy"

// Fetches a value from the blasmatrix `bm` with a bounds check.
// Returns a `mayble<float64>`
pub fun at( bm : blasmatrix, rows : int, cols : int ) : maybe<float64>
  match bm
    Blasmatrix(m-cols, m-rows) ->
      if (cols < 0 || rows < 0) || (cols.int32 >= m-cols || rows.int32 >= m-rows) then
        Nothing
      else
        Just( bm.unsafe-get( cols.ssize_t, rows.ssize_t ) )

// Sets a value in the blasmatrix with `value`.
// WARNING: This will mutate the blasmatrix
pub fun set( bm : blasmatrix, rows : int, cols : int, value : float64 ) : maybe<blasmatrix>
  match bm
    Blasmatrix(m-cols, m-rows, _) ->
      if (cols < 0 || rows < 0) || (cols.int32 >= m-cols || rows.int32 >= m-rows) then
        Nothing
      else
        bm.unsafe-set( rows.ssize_t, cols.ssize_t, value )
        Just( bm )


extern raw-add( ba : blasmatrix, bb : blasmatrix ) : blasmatrix
  c "kk_blasmatrix_add"

pub fun (+)( ba : blasmatrix, bb : blasmatrix ) : maybe<blasmatrix>
  if ba.cols != bb.cols || ba.rows != bb.rows then
    Nothing
  else
    Just( ba.raw-add( bb.copy ) )

fun matrix-tests()
  basic/test("create blasmatrix from matrix<float64>")
    val m = matrix(2, 2, 1.0)
    val bm = m.blasmatrix
    val value = bm.at(0, 0)
    match value
      Nothing -> expect(1, { 0 }, details = "Expected Just(1.0) but found something else" )
      Just(x) -> if x != 1.0 then
          expect(1, { x.int }, details = "Expected Just(1.0) but found something else" )
        else
          ()
  basic/test("create matrix<float64> from blasmatrix")
    val bm = blasmatrix( 2, 2 )
    val m = bm.matrix
    
    val value : maybe<float64> = m.at( 0, 0 )
    match value
      Nothing -> expect(1, { 0 }, details = "Expected Just(0.0) but found something else" )
      Just(x) ->
        if x != 0.0 then
          expect(0, { x.int }, details = "Expected Just(0.0) but found something else" )
        else
          ()