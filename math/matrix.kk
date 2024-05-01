/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// This module provides a matrix which is a column major vector.
module math/matrix
import std/num/float64
import std/num/int32
import std/cextern
import std/test

pub alias matrix<a> = vector<vector<a>>

// Makes an n x m (col x row)) matrix
pub fun matrix( m : int, n : int, default : a ) : matrix<a>
  vector( m, vector( n, default ) )

// Fetches a value from a blasmatrix
pub fun matrix/at( m : matrix<a>, row : int, col : int ) : maybe<a>
  match m.at( row )
    Nothing -> Nothing
    Just(m-row) -> m-row.at( col )

