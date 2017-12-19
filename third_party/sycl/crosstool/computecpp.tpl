#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import tempfile
from subprocess import call, Popen, PIPE, check_output

CPU_CXX_COMPILER = ('%{host_cxx_compiler}')
CPU_C_COMPILER = ('%{host_c_compiler}')
COMPUTECPP_ROOT = ('%{computecpp_root}')

COMPUTECPP_DRIVER = "%s/bin/compute++" % COMPUTECPP_ROOT
COMPUTECPP_INCLUDE = "%s/include" % COMPUTECPP_ROOT

def get_cpp_info(compiler_flags):
  """Check whether the file we are compiling is a CPP file or not."""
  CPP_extensions = ('.cc', '.c++', '.cpp', '.CPP', '.C', '.cxx')
  compiling_cpp = False
  compiled_file_index = None
  compiled_file_name = None

  if '-c' in compiler_flags:
    compiled_file_index = compiler_flags.index('-c') + 1
    compiled_file_name = compiler_flags[compiled_file_index]
    compiling_cpp = compiled_file_name.endswith(CPP_extensions)
  return compiling_cpp, compiled_file_name

def is_external(compiled_file_name, output_file_name):
  """Check if the cfile we are compiling belongs to an external project which
  does not require compiling with ComputeCpp."""
  skip_extensions = [".cu.cc"]
  skip_folders = [
      "tensorflow/compiler",
      "tensorflow/docs_src",
      "tensorflow/stream_executor",
      "tensorflow/tools",
      "third_party",
      "external",
      "hexagon",
      "lite"
  ]
  skip_folders = [(folder + '/') for folder in skip_folders]

  matches_ext = any(compiled_file_name.endswith(_ext) for _ext in skip_extensions)
  matches_folder = any(_folder in output_file_name for _folder in skip_folders)
  return matches_ext or matches_folder


def get_device_compiler_flags(compiler_flags):
  """Remove any flags not used by the device compiler and set up the device
  compiler flags and includes."""
  computecpp_flags = [
      '-isystem', COMPUTECPP_INCLUDE,
      '-Wno-unused-const-variable',
      '-fsycl-ih-last',
      '-sycl-driver',
      '-no-serial-memop',
      '-Xclang', '-cl-denorms-are-zero',
      '-Xclang', '-cl-fp32-correctly-rounded-divide-sqrt',
      '-Xclang', '-cl-mad-enable',
      '-DTENSORFLOW_USE_SYCL=1',
      '-DEIGEN_USE_SYCL=1',
      '-DEIGEN_HAS_C99_MATH',
  ]
  return compiler_flags + computecpp_flags

def checkComputeCppIsSupported():
  outputList = check_output([COMPUTECPP_DRIVER, '--version']).decode('utf-8').split(" ")
  ccpp_version_idx = outputList.index('Device') - 1
  cpp_version = outputList[ccpp_version_idx];
  cppVersionList = cpp_version.split(".")
  if int(cppVersionList[0]) == 0 and int(cppVersionList[1]) < 5:
    print("Error: ComputeCpp {} is not compatible with the current version of Tensorflow, "
          "please update to the latest version of ComputeCpp".format(cpp_version), file=sys.stderr)
    sys.exit(1)

def useDriver(compiler_flags):
  output_file_index = compiler_flags.index('-o') + 1
  output_file_name = compiler_flags[output_file_index]
  if output_file_index == 1:
    # we are linking
    return call([CPU_CXX_COMPILER] + compiler_flags)

  compiling_cpp, compiled_file_name = get_cpp_info(compiler_flags)

  if not compiling_cpp:
    # compile for C
    return call([CPU_C_COMPILER] + compiler_flags)

  if is_external(compiled_file_name, output_file_name):
    return call([CPU_CXX_COMPILER] + compiler_flags)

  filename, file_extension = os.path.splitext(output_file_name)

  computecpp_device_compiler_flags = get_device_compiler_flags(compiler_flags)

  x = call([COMPUTECPP_DRIVER] + computecpp_device_compiler_flags)

  # FIXME(lukeiwanski): throw the sycl line from that dep file
  # that will be fixed in next driver
  dep_file_index = compiler_flags.index('-MF') + 1
  dep_file_name = compiler_flags[dep_file_index]

  f = open(dep_file_name,"r+")
  d = f.readlines()
  f.seek(0)
  for i in d:
      if ".sycl" not in i:
          f.write(i)
  f.truncate()
  f.close()

  return x

def main():
  checkComputeCppIsSupported()
  return useDriver(sys.argv[1:])

if __name__ == '__main__':
  sys.exit(main())
