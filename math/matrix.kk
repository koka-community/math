module math/matrix



pub value struct blasmatrix
  cols : int64
  rows : int64
  internal : intptr_t

pub fun blasmatrix( cols : int64, rows : int64 ) : blasmatrix
  Blasmatrix( cols, rows, allocate-buffer( cols, rows ) )

extern allocate-buffer( cols : int64, rows : int64 ) : intptr_t
  c inline "(long int)kk_malloc(sizeof(double) * #1 * #2, kk_context())"

pub fun blasmatrix/free( bm : blasmatrix ) : ()
  match bm
    Blasmatrix(_, _, internal) ->
      free-internal( internal )

extern free-internal( ptr : intptr_t ) : ()
  c inline "kk_free((void*)#1, kk_context()); return kk_Unit;"

pub alias matrix<a> = vector<vector<a>>

// Makes an n x m (col x row)) matrix
pub fun matrix( n : int, m : int, default : a ) : matrix<a>
  vector( n, vector( m, default ) )
