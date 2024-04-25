/*----------------------------------------------------------------------------
   Copyright 2024, Koka-Community Authors

   Licensed under the MIT License ("The License"). You may not
   use this file except in compliance with the License. A copy of the License
   can be found in the LICENSE file at the root of this distribution.
----------------------------------------------------------------------------*/

// Contains a custom vector type for interacting with BLAS interfaces.
module math/vector
import std/num/int32
import std/num/float64
import std/cextern
import std/core-extras
import std/test

extern import 
  c file "inline/vector"

pub value struct blasvector
  length : int32
  internal : owned-c<float64>


pub fun blasvector( len : int ) : blasvector
  Blasvector( len.int32, c-own( allocate-buffer( len.int32 ) ) )
  
extern allocate-buffer( len : int32 ) : intptr_t
  c inline "(long int)kk_malloc(sizeof(double) * #1, kk_context())"

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

pub fun at( bv : blasvector, index : int ) : maybe<float64>
  if index < 0 || index.int32 >= bv.length then
    Nothing
  else
    Just( bv.unsafe-get( index.ssize_t ) )

// WARNING: This will mutate the blasvector
pub fun set( bv : blasvector, index : int, value : float64 ) : maybe<blasvector>
  if index < 0 || index.int32 >= bv.length then
    Nothing
  else
    bv.unsafe-set( index.ssize_t, value )
    Just( bv )

fun vector-tests()
  basic/test("create blasvector from vector<float64>")
    val v = vector( 2, 1.0 )
    val bv = v.blasvector
    val value : maybe<float64> = bv.at( 0 )
    match value
      Nothing -> expect(1, { 0 }, details = "Expected Just(1.0) but found something else" )
      Just(x) -> if x != 1.0 then
          expect(1, { x.int }, details = "Expected Just(1.0) but found something else" )
        else
          ()
  basic/test("create vector<float64> from blasvector")
    val bv = blasvector( 2 )
    val v = bv.vector
    val value : maybe<float64> = v.at( 0 )
    match value
      Nothing -> expect(1, { 0 }, details = "Expected Just(0.0) but found something else" )
      Just(x) ->
        if x != 0.0 then
          expect(0, { x.int }, details = "Expected Just(0.0) but found something else" )
        else
          ()
