module math/matrix
import std/num/int32
import std/cextern


pub value struct blasmatrix
  cols : int32
  rows : int32
  internal : owned-c<float64>

pub fun blasmatrix( cols : int32, rows : int32 ) : blasmatrix
  Blasmatrix( cols, rows, c-own( allocate-buffer( cols, rows ) ) )

extern allocate-buffer( cols : int32, rows : int32 ) : intptr_t
  c inline "(long int)malloc(sizeof(double) * #1 * #2)"


pub alias matrix<a> = vector<vector<a>>

// Makes an n x m (col x row)) matrix
pub fun matrix( n : int, m : int, default : a ) : matrix<a>
  vector( n, vector( m, default ) )
