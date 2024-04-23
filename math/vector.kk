/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// Contains a custom vector type for interacting with BLAS interfaces.
module math/vector
import std/num/int32
import std/cextern

extern import 
  c file "inline/vector"

pub value struct blasvector
  length : int32
  internal : owned-c<float64>


pub fun blasvector( len : int32 ) : blasvector
  Blasvector( len, c-own( allocate-buffer( len ) ) )
  
extern allocate-buffer( len : int32 ) : intptr_t
  c inline "(long int)malloc(sizeof(double) * #1)"

pub extern vector/blasvector( v : vector<float64> ) : blasvector
  c "kk_vector_blasvector"

// Note: this function will free the blasvector
pub extern blasvector/vector( bv : blasvector ) : vector<float64>
  c "kk_blasvector_vector"