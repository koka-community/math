module math/matrix

pub alias matrix<a> = vector<vector<a>>

// Makes an n x m (col x row)) matrix
pub fun matrix( n : int, m : int, default : a ) : matrix<a>
  vector( n, vector( m, default ) )
