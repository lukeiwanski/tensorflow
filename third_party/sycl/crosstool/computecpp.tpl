#!/usr/bin/env python

import os
import sys
import tempfile
from subprocess import call, Popen, PIPE

CPU_CXX_COMPILER = ('%{host_cxx_compiler}')
CPU_C_COMPILER = ('%{host_c_compiler}')

CURRENT_DIR = os.path.dirname(sys.argv[0])
COMPUTECPP_ROOT = CURRENT_DIR + '/../sycl/'
COMPUTECPP_DRIVER= COMPUTECPP_ROOT + 'bin/compute++'
COMPUTECPP_INCLUDE = COMPUTECPP_ROOT + 'include'

def clean_passed_in_flags():
  """Extract flags from args and remove unwanted options."""
  remove_flags = ('-Wl,--no-undefined', '-Wno-unused-but-set-variable', '-Wignored-attributes')
  # remove -fsanitize-coverage from string with g++
  if 'g++' in CPU_CXX_COMPILER:
    remove_flags += ('-fsanitize-coverage',)
  compiler_flags = [flag for flag in sys.argv[1:] if not flag.startswith(remove_flags)]
  return compiler_flags

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
  return compiling_cpp, compiled_file_index, compiled_file_name

def add_env_vars_to_flags(flags):
  """Add required env vars to compiler flags as needed for all compilers."""
  # add -D_GLIBCXX_USE_CXX11_ABI=0 to the command line if you have custom installation of GCC/Clang
  extra_env_vars = [
      '-DEIGEN_HAS_C99_MATH',
  ]
  return flags + extra_env_vars

def add_sycl_env_vars_to_flags(flags):
  """Add required env vars to compiler flags as needed for ComputeCpp."""
  extra_env_vars = [
      '-DTENSORFLOW_USE_SYCL=1',
      '-DEIGEN_USE_SYCL=1'
  ]
  # If TF_VECTORIZE_SYCL is defined and positive, don't add the flag to disable
  # vectorisation
  if int(os.getenv('TF_VECTORIZE_SYCL', -1)) <= 0:
    extra_env_vars += ['-DEIGEN_DONT_VECTORIZE_SYCL=1']
  return flags + extra_env_vars

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
      "hexagon"
  ]
  skip_folders = [(folder + '/') for folder in skip_folders]

  matches_ext = any(compiled_file_name.endswith(_ext) for _ext in skip_extensions)
  matches_folder = any(_folder in output_file_name for _folder in skip_folders)
  return matches_ext or matches_folder

def get_preprocessor_flags(compiler_flags, output_file_index):
  """Set up the preprocessor flags, removing any output flags."""
  flags_without_output = list(compiler_flags)
  del flags_without_output[output_file_index]   # remove output_file_name
  del flags_without_output[output_file_index - 1] # remove '-o'
  return flags_without_output + ['-E']

def remove_unknown_computecpp_flags(flags):
  """Remove flags not recognised by the device compiler. Also remove any CPU
  specific vectorization flags as they are not applicable to SYCL devices."""
  unknown_flags = [
      '-fsanitize',
      '-fno-canonical-system-headers',
      '=native',
      '=core2',
      'msse',
      'mavx',
      'mmmx',
      'm3dnow',
      'fma'
  ]
  return [flag for flag in flags if not any(x in flag.lower() for x in unknown_flags)]

def get_device_compiler_flags(compiler_flags):
  """Remove any flags not used by the device compiler and set up the device
  compiler flags and includes."""
  computecpp_flags = [
      '-sycl-compress-name',
      '-Wno-unused-variable',
      '-Wno-c++11-narrowing',
      '-I', COMPUTECPP_INCLUDE,
      '-isystem', COMPUTECPP_INCLUDE,
      '-std=c++11',
      '-sycl', '-emit-llvm', '-no-serial-memop',
      '-Xclang', '-cl-denorms-are-zero',
      '-Xclang', '-cl-fp32-correctly-rounded-divide-sqrt',
      '-Xclang', '-cl-mad-enable',
      '-Xclang', '-cl-unsafe-math-optimizations'
  ]
  return computecpp_flags + remove_unknown_computecpp_flags(compiler_flags)

def get_host_compiler_flags(compiler_flags, bc_out):
  """Remove device compiler specific flags and set up the host compiler flags."""
  device_flag_prefixes = ('-MF', '-MD',)
  host_compiler_flags = [flag for flag in compiler_flags if (not flag.startswith(device_flag_prefixes) and not '.d' in flag)]
  host_compiler_flags[host_compiler_flags.index('-c')] = "--include"
  host_compiler_flags = [
      '-xc++',
      '-Wno-unused-variable',
      '-I', COMPUTECPP_INCLUDE,
      '-c', bc_out
  ] + host_compiler_flags
  return host_compiler_flags

def main():
  compiler_flags = clean_passed_in_flags()

  output_file_index = compiler_flags.index('-o') + 1
  output_file_name = compiler_flags[output_file_index]

  if output_file_index == 1:
    # we are linking
    return call([CPU_CXX_COMPILER] + compiler_flags + ['-Wl,--no-undefined'])

  compiler_flags = add_env_vars_to_flags(compiler_flags)
  compiling_cpp, compiled_file_index, compiled_file_name = get_cpp_info(compiler_flags)

  if not compiling_cpp:
    # compile for C
    return call([CPU_C_COMPILER] + compiler_flags)

  if is_external(compiled_file_name, output_file_name):
    return call([CPU_CXX_COMPILER] + compiler_flags)

  compiler_flags = add_sycl_env_vars_to_flags(compiler_flags)

  # this is an optimisation that will check if compiled file has to be compiled
  # with ComputeCpp. Any file without a 'parallel_for' can be skipped and
  # compiled solely with the host compiler.
  pp_flags = get_preprocessor_flags(compiler_flags, output_file_index)
  pipe = Popen([CPU_CXX_COMPILER] + pp_flags, stdout=PIPE)
  preprocessed_file_str = pipe.communicate()[0]
  if pipe.returncode != 0:
    return pipe.returncode

  # check if it has parallel_for in it
  if not b'.parallel_for' in preprocessed_file_str:
    # call CXX compiler like usual
    # Force '.ii' extension so that g++ does not preprocess the file again
    with tempfile.NamedTemporaryFile(suffix=".ii") as preprocessed_file:
      preprocessed_file.write(preprocessed_file_str)
      preprocessed_file.flush()
      compiler_flags[compiled_file_index] = preprocessed_file.name
      return call([CPU_CXX_COMPILER] + compiler_flags)
  del preprocessed_file_str   # save some memory as this string can be quite big

  filename, file_extension = os.path.splitext(output_file_name)
  bc_out = filename + '.sycl'

  computecpp_device_compiler_flags = get_device_compiler_flags(compiler_flags)

  x = call([COMPUTECPP_DRIVER] + computecpp_device_compiler_flags)
  if x == 0:
    host_compiler_flags = get_host_compiler_flags(compiler_flags, bc_out)
    x = call([CPU_CXX_COMPILER] + host_compiler_flags)
  return x

if __name__ == '__main__':
  sys.exit(main())
