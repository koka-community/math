/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// This module provides a matrix which is a column major vector alias and a blasvector which is used for interacting with BLAS APIs.
module math/matrix
import std/num/float64
import std/num/int32
import std/cextern
import std/test

extern import
  c file "inline/matrix"

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

pub alias matrix<a> = vector<vector<a>>

// Makes an n x m (col x row)) matrix
pub fun matrix( m : int, n : int, default : a ) : matrix<a>
  vector( m, vector( n, default ) )

// Fetches a value from a blasmatrix
pub fun matrix/at( m : matrix<a>, row : int, col : int ) : maybe<a>
  match m.at( row )
    Nothing -> Nothing
    Just(m-row) -> m-row.at( col )

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

fun main()
  val ba = blasmatrix(2, 2)
  ba.unsafe-set( 0.ssize_t, 0.ssize_t, 0.0)
  ba.unsafe-set( 1.ssize_t, 0.ssize_t, 0.0)
  ba.unsafe-set( 0.ssize_t, 1.ssize_t, 0.0)
  ba.unsafe-set( 1.ssize_t, 1.ssize_t, 3.0)
  val bb = blasmatrix(2,2)
  bb.unsafe-set( 0.ssize_t, 0.ssize_t, 0.0)
  bb.unsafe-set( 1.ssize_t, 0.ssize_t, 0.0)
  bb.unsafe-set( 0.ssize_t, 1.ssize_t, 0.0)
  bb.unsafe-set( 1.ssize_t, 1.ssize_t, 3.0)

  val bout = match ba + bb
    Just(x) -> x
    Nothing -> bb
  for(2) fn (i)
    for(2) fn (j)
      print(ba.unsafe-get(i.ssize_t, j.ssize_t).show ++ " ")
    println("")
  println("")
  for(2) fn (i)
    for(2) fn (j)
      print(bb.unsafe-get(i.ssize_t, j.ssize_t).show ++ " ")
    println("")
  println("")
  for(2) fn (i)
    for(2) fn (j)
      print(bout.unsafe-get(i.ssize_t, j.ssize_t).show ++ " ")
    println("")
