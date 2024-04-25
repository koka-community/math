module math/matrix
import std/num/float64
import std/num/int32
import std/cextern
import std/test

extern import
  c file "inline/matrix"

pub value struct blasmatrix
  cols : int32
  rows : int32
  internal : owned-c<float64>

pub fun blasmatrix( cols : int, rows : int ) : blasmatrix
  Blasmatrix( cols.int32, rows.int32, c-own( allocate-buffer( cols.int32, rows.int32 ) ) )

extern allocate-buffer( cols : int32, rows : int32 ) : intptr_t
  c inline "(long int)kk_malloc(sizeof(double) * #1 * #2, kk_context())"

pub extern matrix/blasmatrix( m : matrix<float64> ) : blasmatrix
  c "kk_matrix_blasmatrix"

pub extern blasmatrix/matrix( bm : blasmatrix ) : matrix<float64>
  c "kk_blasmatrix_matrix"


inline extern unsafe-get( bm : blasmatrix, col : ssize_t, row : ssize_t ) : float64
  c "kk_blasmatrix_unsafe_get"

inline extern unsafe-set( bm : blasmatrix, col : ssize_t, row : ssize_t, value : float64 ) : ()
  c "kk_blasmatrix_unsafe_set"

pub extern copy( bv : blasmatrix ) : blasmatrix
  c "kk_blasmatrix_copy"

pub fun at( bm : blasmatrix, cols : int, rows : int ) : maybe<float64>
  match bm
    Blasmatrix(m-cols, m-rows) ->
      if (cols < 0 || rows < 0) || (cols.int32 >= m-cols || rows.int32 >= m-rows) then
        Nothing
      else
        Just( bm.unsafe-get( cols.ssize_t, rows.ssize_t ) )

pub fun set( bm : blasmatrix, cols : int, rows : int, value : float64 ) : maybe<blasmatrix>
  match bm
    Blasmatrix(m-cols, m-rows) ->
      if (cols < 0 || rows < 0) || (cols.int32 >= m-cols || rows.int32 >= m-rows) then
        Nothing
      else
        bm.unsafe-set( cols.ssize_t, rows.ssize_t, value )
        Just( bm )

pub alias matrix<a> = vector<vector<a>>

// Makes an n x m (col x row)) matrix
pub fun matrix( n : int, m : int, default : a ) : matrix<a>
  vector( n, vector( m, default ) )

pub fun matrix/at( m : matrix<a>, col : int, row : int ) : maybe<a>
  match m.at( col )
    Nothing -> Nothing
    Just(m-row) -> m-row.at( row )

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