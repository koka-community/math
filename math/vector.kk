/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// Contains a custom vector type for interacting with BLAS interfaces.
module math/vector

extern import 
  c file "inline/vector"

pub value struct blasvector
  length : int64
  internal : intptr_t


pub fun blasvector( len : int64 ) : blasvector
  Blasvector( len, allocate_buffer( len ))
  
extern allocate_buffer( len : int64 ) : intptr_t
  c inline "(long int)kk_malloc(sizeof(double) * #1, kk_context())"

pub extern vector/blasvector( v : vector<float64> ) : blasvector
  c "kk_vector_blasvector"

pub fun blasvector/free( bv : blasvector ) : ()
  match bv
    Blasvector(_, internal) ->
      free-internal( internal )

extern free-internal( ptr : intptr_t ) : ()
  c inline "kk_free((void*)#1, kk_context()); return kk_Unit;"

// Note: this function will free the blasvector
pub extern blasvector/vector( bv : blasvector ) : vector<float64>
  c "kk_blasvector_vector"