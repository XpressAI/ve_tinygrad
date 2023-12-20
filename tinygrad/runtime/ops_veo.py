import ctypes, subprocess, functools, pathlib, tempfile
from typing import Any, Union
from tinygrad.helpers import dtypes, flat_mv, DEBUG
from tinygrad.device import Compiled, Allocator, MallocAllocator, LRUAllocator
from tinygrad.helpers import diskcache, cpu_time_execution
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
import sys
import veo
import os
import time
import hashlib
import math
import re

from tinygrad.renderer.llvmir import uops_to_llvm_ir

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half float\n#define uchar unsigned char\n#include <stdbool.h>\n'

def ve_time_execution(cb, enable):
  if enable: st = time.perf_counter()
  res = cb()
  res.wait_result()
  if enable: return time.perf_counter()-st


def rewrite_to_no_acc(input_code):
  # find eventual target
  target_regex = re.compile(r"(data0\[.+?\]) = acc0;")
  match = target_regex.search(input_code)
  if match is None:
    return input_code

  target = match.group(1)

  init_regex = re.compile(r"float acc0 =")

  acc_regex = re.compile(r"acc0 = \((.+?)([-+*/])acc0\);")

  without_final_set = target_regex.sub("", input_code)
  with_init = init_regex.sub(f"{target} =", without_final_set)
  acc_match = acc_regex.search(with_init)
  if acc_match is None:
    return input_code
  with_inplace_update = acc_regex.sub(f"{target} {acc_match.group(2)}= {acc_match.group(1)};", with_init)

  return with_inplace_update

@diskcache
def compile_ve_lib(prg:str, header:str=CLANG_PROGRAM_HEADER):
  code = header + prg + '\n'
  label = re.search(r"void (.+)\(", prg).group(1)
  builder = veo.VeBuild()
  builder.set_c_src(label, code, flags=' '.join([
      '-shared',
      '-fpic',
      '-O4',
      '-fopenmp',
      '-mparallel',
      #'-mvector-packed',
       #'-finline-functions',
       #'-fno-defer-inline-template-instantiation',
       #'-finline-max-depth=10',
       '-ffast-math',
       '-msched-block',
       '-report-all',
       '-fdiag-vector=2',
    #'-floop-unroll-complete=512',
    #'-mvector-advance-gather-limit=512',
    #'-fno-strict-aliasing',
    #'-fouterloop-unroll',
    '-fmatrix-multiply',
    #'-floop-strip-mine',
    #'-floop-normalize',
    #'-floop-interchange',
    #'-floop-fusion',
    #'-floop-collapse',

    # '-mcreate-threads-at-startup',
    # '-finline-functions',
    # '-fno-defer-inline-template-instantiation',
    # '-finline-max-depth=10',
    # '-ffast-math',
    # '-fivdep',
    # '-floop-collapse',
    # '-floop-fusion',
    # '-floop-interchange',
    # '-floop-normalize',
    # '-floop-split',
    # '-floop-strip-mine',
    # '-floop-unroll',
     '-fassociative-math',
    # '-faggressive-associative-math',
    # '-fmatrix-multiply',
    # '-fmove-loop-invariants-unsafe',
    # '-mlist-vector',
    # '-fnaked-ivdep',
    # # '-msched-interblock',
    # '-mvector-floating-divide-instruction',
    # '-mvector-low-precise-divide-function',
    # '-mvector-merge-conditional',
    # '-mvector-power-to-explog',
    # '-mvector-power-to-sqrt',
    # '-mvector-shortloop-reduction',
    # '-mvector-sqrt-instruction',
    # # '-mparallel-innerloop',
    # # '-mparallel-outerloop-strip-mine',
    # # '-mparallel-sections',
    # # '-mschedule-dynamic',
    # '-mparallel',
    # '-report-all',
    # '-fdiag-vector=2'

  ]))
  veo_name = builder.build_so(label)

  builder.clean()
  return os.getcwd() + "/" + veo_name

class VEProgram:
  def __init__(self, device, name:str, lib):
    self.device = device
    self.ve_lib = device.proc.load_library(str.encode(lib))
    self.fn_name = name
    self.fxn = None

  def __call__(self, *bufs, vals=(), wait=False): 
    #print(f"bufs = {bufs}")
    #print(f"vals = {vals}")

    if self.fxn is None:
      self.fxn = getattr(self.ve_lib, self.fn_name)
      self.fxn.ret_type('void')
      self.fxn.args_type(*(['double *' for _ in bufs] + ['unsigned int' for _ in vals]))
    return ve_time_execution(lambda: self.fxn(self.device.ctx, *bufs, *vals), wait)
    #return self.fxn(*[veo.OnStack(buf, inout=veo.INTENT_INOUT) for buf in bufs], *vals, sync=False)


class VeoAllocator(LRUAllocator):
  def __init__(self, device):
    self.device = device
    super().__init__()

  def _alloc(self, size:int):
    mem = self.device.proc.alloc_mem(int(8 * math.ceil(size / 8)))
    return mem

  def copyin(self, dest, src:memoryview):
    self.device.ctx.async_write_mem(dest, src, len(src))

  def copyout(self, dest:memoryview, src):
    self.device.proc.read_mem(dest, src, len(dest))

  def _free(self, opaque):
    self.device.proc.free_mem(opaque)

class VELanguage(CStyleLanguage):
  uses_ptr_arithmetic=False

  arg_int_prefix = "const int"
  buffer_suffix = " restrict"
  def render_for(self, expr: str, _min:Union[int,str], _max:Union[int,str]) -> str:
    out = ""
    return out+f"for (int {expr} = {_min}; {expr} < {_max}; ++{expr}) {{"

def renderer(function_name, ast):
  code = uops_to_cstyle(VELanguage(), function_name, ast)
  return (rewrite_to_no_acc(code[0]), code[1])

class VEODevice(Compiled):
  def __init__(self, device):
    self.node_number = int(os.environ.get('VE_NODE_NUMBER') or 0)
    self.proc = veo.VeoProc(self.node_number)
    self.ctx = self.proc.open_context()
    super().__init__(VeoAllocator(self), LinearizerOptions(supports_float4=False, has_local=False, device="VEO"), renderer, compile_ve_lib, functools.partial(VEProgram, self))

  def __del__(self):
    self.proc.close_context(self.ctx)
    self.proc.proc_destroy()


  def get_linearizer(self, ast):
    if DEBUG >= 3:
      from tinygrad.graph import print_tree
      print_tree(ast)
    from tinygrad.codegen.linearizer import Linearizer
    k = Linearizer(ast, self.linearizer_opts)
    k.required_optimizations()
    return k

