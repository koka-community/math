module math/matrix

pub alias matrix<a> = vector<vector<a>>

// Makes an m x n (row x col) matrix
pub fun matrix( m : int, n : int, default : a ) : matrix<a>
  vector( m, vector( n, default ) )
