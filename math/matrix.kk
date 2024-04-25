module math/matrix
import std/num/int32
import std/cextern

extern import
  c file "inline/matrix"

pub value struct blasmatrix
  cols : int32
  rows : int32
  internal : owned-c<float64>

pub fun blasmatrix( cols : int32, rows : int32 ) : blasmatrix
  Blasmatrix( cols, rows, c-own( allocate-buffer( cols, rows ) ) )

extern allocate-buffer( cols : int32, rows : int32 ) : intptr_t
  c inline "(long int)malloc(sizeof(double) * #1 * #2)"

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

pub fun get( bm : blasmatrix, cols : int, rows : int ) : maybe<float64>
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

fun main() 
  val m = matrix(2, 2, 1.0)
  val bm = m.blasmatrix
  ()