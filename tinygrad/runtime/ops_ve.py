import numpy as np
import nlcpy as vp
from typing import Callable, Dict, Tuple
from tinygrad.helpers import dtypes, flat_mv
from tinygrad.ops import BufferOps, UnaryOps, BinaryOps, MovementOps, ReduceOps, TernaryOps, Op
from tinygrad.device import Interpreted, Allocator

vp.request.set_offload_timing_onthefly(venode=vp.venode.VE(0))


def shape_to_axis(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]) -> Tuple[int, ...]:
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

# TODO: this should be global infrastructure
def output_type(x, y): return x.dtype if dtypes.from_np(x.dtype).priority > dtypes.from_np(y.dtype).priority else y.dtype
def match_types(x, y):
  up = output_type(x, y)
  return x.astype(up, copy=False), y.astype(up, copy=False)

def einsum_mulacc(einsum, get_strides, expand):
  def einscripts(x): return ''.join(["abcdefghijklmnopqrstuvwxyz"[i] for i in x])
  def axes_slice(strides): return [i for i,s in enumerate(strides) if s != 0], tuple([slice(None) if s != 0 else 0 for i,s in enumerate(strides)])
  def mulacc(a, b, new_shape):
    (a_axes, a_slices), (b_axes, b_slices) = axes_slice(get_strides(a)), axes_slice(get_strides(b))
    out = [i for i in range(len(new_shape)) if a.shape[i] == new_shape[i] and (i in a_axes or i in b_axes)]
    ret = einsum(f"{einscripts(a_axes)}, {einscripts(b_axes)} -> {einscripts(out)}", a[a_slices], b[b_slices])
    return expand(ret.reshape([(1 if i not in a_axes and i not in b_axes else s) for i,s in enumerate(new_shape)]), new_shape)
  return mulacc

def broadcast_to(array, shape, subok=False):
  if array.dtype == np.uint8:
    return np.broadcast_to(array, shape, subok)
  else:
    return vp.broadcast_to(array, shape, subok)

def print_return(s):
  #print(s)
  return s

def my_einsum(s, A, B, optimize=True):
    # Split the input string into input and output parts
    input_str, output_str = s.split('->')
    input_str = input_str.replace(' ', '')  # Remove any spaces
    inputs = input_str.split(',')

    # Make sure we have two inputs
    assert len(inputs) == 2, "Function only accepts two input operands."

    # Create a dictionary for mapping the letters to array dimensions
    letter_dims = {}
    next_dim = 0
    for letter in inputs[0] + inputs[1]:
        if letter not in letter_dims:
            letter_dims[letter] = next_dim
            next_dim += 1

    # Reshape A and B according to the letters in the inputs
    A_shape = [1] * next_dim
    B_shape = [1] * next_dim
    for i, letter in enumerate(inputs[0]):
        A_shape[letter_dims[letter]] = A.shape[i]
    for i, letter in enumerate(inputs[1]):
        B_shape[letter_dims[letter]] = B.shape[i]
    A_reshaped = A.reshape(A_shape)
    B_reshaped = B.reshape(B_shape)

    # Perform the multiplication
    result = A_reshaped * B_reshaped

    # Sum over the axes that are not in the output
    for letter in letter_dims:
        if letter not in output_str:
            axis = letter_dims[letter]
            result = result.sum(axis=axis, keepdims=True)

    # Squeeze to remove dimensions of size one which are not in the output
    result_shape = [result.shape[letter_dims[letter]] for letter in output_str if letter in letter_dims]
    result = result.reshape(result_shape)

    return result

numpy_fxn_for_op: Dict[Op, Callable] = {
  BufferOps.CONST: lambda val, dtype: vp.array(val, dtype=dtype.np),
  UnaryOps.EXP2: vp.exp2, 
  UnaryOps.LOG2: vp.log2, 
  UnaryOps.SIN: vp.sin,
  UnaryOps.CAST: lambda x,y: x.view(y[0].np) if y[1] else x.astype(y[0].np, copy=False),
  UnaryOps.NEG: lambda x: vp.logical_not(x) if x.dtype == vp.bool_ else vp.negative(x),
  BinaryOps.MAX: vp.maximum, 
  BinaryOps.CMPLT: lambda x,y: (x<y).astype(output_type(x,y)), 
  BinaryOps.ADD: lambda x, y: vp.add(*match_types(x, y)),
  BinaryOps.SUB: lambda x, y: vp.subtract(*match_types(x, y)), 
  BinaryOps.MUL: lambda x, y: vp.multiply(*match_types(x, y)),
  BinaryOps.DIV: lambda x, y: vp.divide(*match_types(x, y)).astype(output_type(x, y), copy=False), 
  BinaryOps.XOR: lambda x, y: vp.bitwise_xor(*match_types(x, y)), 
  UnaryOps.SQRT: vp.sqrt,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), dtype=x.dtype, keepdims=True) if x.shape != new_shape else x,
  ReduceOps.MAX: lambda x, new_shape: x.max(shape_to_axis(x.shape, new_shape), keepdims=True) if x.shape != new_shape else x,
  MovementOps.AS_STRIDED: lambda x, arg: np.ndarray(arg[0], buffer=np.require(x, requirements='C'), dtype=x.dtype, offset=arg[2]*x.dtype.itemsize, strides=tuple(y*x.dtype.itemsize for y in arg[1])),
  MovementOps.PAD: np.pad, 
  MovementOps.EXPAND: broadcast_to,
  TernaryOps.MULACC: einsum_mulacc(lambda s,a,b: my_einsum(print_return(s), *match_types(a.copy(), b.copy()), optimize=True), lambda x: x.strides, broadcast_to),
  TernaryOps.WHERE: vp.where,
}

class NLCPyAllocator(Allocator):
  def _alloc(self, size:int): 
    return vp.empty(size // 4, dtype=vp.float32)

  def copyin(self, dest: vp.ndarray, src:memoryview): 
      vp.copyto(dest, np.frombuffer(src, dest.dtype).reshape(dest.shape))
      #vp.venode.synchronize_all_ve()

  def copyout(self, dest:memoryview, src:vp.ndarray): 
      np.copyto(np.frombuffer(dest, src.dtype).reshape(src.shape), src)
      #vp.venode.synchronize_all_ve()


class VEDevice(Interpreted):
  def __init__(self, device):
    super().__init__(NLCPyAllocator(), numpy_fxn_for_op)

  def synchronize(self):
    print("wee")
    vp.venode.synchronize_all_ve()
