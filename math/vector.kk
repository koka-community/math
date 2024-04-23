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
import std/core-extras

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

pub extern blasvector/vector( bv : blasvector ) : vector<float64>
  c "kk_blasvector_vector"

// Get a value from a blasvector without a bounds check
inline extern unsafe-get( bv : blasvector, index : ssize_t ) : float64
  c "kk_blasvector_unsafe_get"

// Set a value in the blasvector without a bounds check
inline extern unsafe-set( bv : blasvector, index : ssize_t, value : float64 ) : ()
  c "kk_blasvector_unsafe_set"

pub extern copy( bv : blasvector ) : blasvector
  c "kk_blasvector_copy"

pub fun get( bv : blasvector, index : int ) : maybe<float64>
  if index < 0 || index.int32 >= bv.length then
    Nothing
  else
    Just( bv.unsafe-get( index.ssize_t ) )

// WARNING: This will mutate the vector
pub fun set( bv : blasvector, index : int, value : float64 ) : maybe<blasvector>
  if index < 0 || index.int32 >= bv.length then
    Nothing
  else
    bv.unsafe-set( index.ssize_t, value )
    Just( bv )