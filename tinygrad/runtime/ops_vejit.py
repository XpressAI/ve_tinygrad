import ctypes, subprocess, functools, pathlib, tempfile
from typing import Any
from tinygrad.helpers import dtypes, flat_mv
from tinygrad.device import Compiled, Allocator, MallocAllocator
from tinygrad.helpers import diskcache, cpu_time_execution
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
import nlcpy
import nlcpy as vp
import numpy as np
from nlcpy import ve_types
from .ops_ve import NLCPyAllocator
import sys
from nlcpy import veo

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half float\n#define uchar unsigned char\n#include <stdbool.h>\n'

def compile_ve_lib(prg:str, header:str=CLANG_PROGRAM_HEADER):
  code = header + prg + '\n'

  #print(code)

  ve_lib = nlcpy.jit.CustomVELibrary(
    code=code,
    cflags=nlcpy.jit.get_default_cflags(openmp=False, opt_level=4) + (
      '-mvector-packed', 
      '-finline-functions',
      '-fno-defer-inline-template-instantiation', 
      '-finline-max-depth=10',
      '-ffast-math',
      '-msched-block',
      '-report-all',
      '-fdiag-vector=2'
    ),
    log_stream=sys.stdout
  )

  return ve_lib

class VEProgram:
  def __init__(self, name:str, lib):
    self.ve_lib = lib
    self.fn_name = name
    self.fxn = None

  def __call__(self, *bufs, vals=(), wait=False): 
    #print(f"bufs = {bufs}")
    #print(f"vals = {vals}")

    if self.fxn is None:
      self.fxn = self.ve_lib.get_function(
        self.fn_name, 
        args_type=tuple(ve_types.uint64 for x in bufs) + tuple(ve_types.uint64 for x in vals),
        ret_type=ve_types.void
      )
    
    return self.fxn(*[buf.ve_adr for buf in bufs], *vals, sync=False)
    #return self.fxn(*[veo.OnStack(buf, inout=veo.INTENT_INOUT) for buf in bufs], *vals, sync=False)

class VEJitAllocator(Allocator):
  def _alloc(self, size:int): 
    return vp.empty(size // 4, dtype=vp.float32)

  def copyin(self, dest: vp.ndarray, src:memoryview): 
    vp.copyto(dest, np.frombuffer(src, dest.dtype).reshape(dest.shape))
    #vp.venode.synchronize_all_ve()

  def copyout(self, dest:memoryview, src:vp.ndarray): 
    np.copyto(np.frombuffer(dest, src.dtype).reshape(src.shape), src)
    #vp.venode.synchronize_all_ve()

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(buffer_suffix=" restrict", arg_int_prefix="const int"))
VEJITDevice = Compiled(VEJitAllocator(), LinearizerOptions(supports_float4=False, has_local=False), renderer, compile_ve_lib, VEProgram)
#VEJITDevice = Compiled(MallocAllocator, LinearizerOptions(supports_float4=False, has_local=False), renderer, compile_ve_lib, VEProgram)
