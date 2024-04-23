// Bindings for OpenBLAS for Koka
// To compile make sure that you set the linker flags: -lm -lpthread gfortran -lopenblas
// Here is a snippet to help: `--cclinkopts=-lm --cclinkopts=-lpthread --cclinkopts=-lgfortran --cclinkopts=-lopenblas`
module math/linalg
import std/test
import std/num/float64
import std/core-extras
import math/matrix

extern import 
  c file "inline/linalg"


// Performs a dot product between two vectors of `float64`.
// Note: the smallest vector will be the maximum size for this operation
pub extern vector/dot( a : vector<float64>, b : vector<float64> ) : float64
  c "kk_dot_product"

// Performs the Euclidean norm of a vector.
pub extern vector/norm( v : vector<float64> ) : float64
  c "kk_vector_norm"

// Performs rotation of points in the plane.
// Note: the smallest vector will be the maximum size for this operation
pub fun vector/rotation( a : vector<float64>, b : vector<float64>, c : float64 = 1.0, s : float64 = 1.0 ) : (vector<float64>, vector<float64>)
  val v = vector/rotation-inline( a, b, c, s )
  (v.unsafe-idx( 0.ssize_t ), v.unsafe-idx( 1.ssize_t ))

inline extern vector/rotation-inline( a : vector<float64>, b : vector<float64>, c : float64, s : float64 ) : vector<vector<float64>>
  c "kk_vector_rotation"

// Computes the parameters for a Givens rotation
// Returns: (r, z, c, s)
// Note: the smallest vector will be the maximum size for this operation
pub fun vector/givens-rotation-params( x : float64, y: float64 ) : (float64, float64, float64, float64)
  val v = vector/givens-rotation-params-inline(x, y)
  (v.unsafe-idx( 0.ssize_t ), v.unsafe-idx( 1.ssize_t ), v.unsafe-idx( 2.ssize_t ), v.unsafe-idx( 3.ssize_t ))

inline extern vector/givens-rotation-params-inline( x : float64, y : float64) : vector<float64>
  c "kk_vector_rotation_givens_params"

pub fun vector/modified-givens-rotation( x : vector<float64>, y : vector<float64>, h : matrix<float64> ) : (vector<float64>, vector<float64>)
  val v = vector/modified-givens-rotation-inline( x, y, h )
  (v.unsafe-idx( 0.ssize_t ), v.unsafe-idx( 1.ssize_t ))

inline extern vector/modified-givens-rotation-inline( x : vector<float64>, y : vector<float64>, h : matrix<float64> ) : vector<vector<float64>>
  c "kk_modified_givens_rotation"

// Scales a vector by a given scalar.
pub extern vector/scalar-product( v : vector<float64>, scalar : float64 ) : vector<float64>
  c "kk_scale_vector"

// Computes the sum of magnitudes of the vector elements.
pub extern vector/sum-of-magnitudes( v : vector<float64> ) : float64
  c "kk_sum_magnitudes"


pub fun linalg-tests()
  basic/test("dot product of two vectors")
    val x = ([1, 2, 3].map fn (i) i.float64).vector
    val y = ([1, 2, 3].map fn (i) i.float64).vector
    val value = x.dot( y )
    expect(14, { value.int }, details = "Expected 14 but got " ++ value.int.show)
  basic/test("scaling vector")
    val v = ([1, 2, 3].map fn (i) i.float64).vector
    val scalar = 2.float64
    val v' = v.scalar-product(scalar)
    var buffer := ""
    v'.foreach fn (f)
      buffer := buffer ++ f.show ++ " "
    expect("2 4 6 ", { buffer }, details = "Expected '2 4 6 ' but got " ++ buffer)

fun main()
  linalg-tests()
  val x = ([1, 2, 3].map fn (i) i.float64).vector
  val y = ([1, 2, 3].map fn (i) i.float64).vector
  val (a, b) = x.rotation(y)
  var buffer1 := ""
  var buffer2 := ""
  a.foreach fn (f)
      buffer1 := buffer1 ++ f.show ++ " "
  b.foreach fn (f)
      buffer2 := buffer2 ++ f.show ++ " "
  buffer1.println
  buffer2.println
