# SPDX-License-Identifier: AGPL-3.0-or-later
# Compile-only acceptance of exact optional MMA forms on each requested target.
# Optional disassembly distinguishes direct MMA from compiler expansions.
# Neither compilation nor disassembly executes code or proves runtime support.
function(archbench_probe_mma arch probe_name output_variable)
  set(probe_root "${CMAKE_BINARY_DIR}/capability-probes/sm_${arch}")
  file(MAKE_DIRECTORY "${probe_root}")
  set(${output_variable}_CODEGEN "not_inspected" PARENT_SCOPE)
  if(NOT ARCHBENCH_COMPILE_CAPABILITY_PROBES)
    file(WRITE "${probe_root}/${probe_name}.log" "Not probed: ARCHBENCH_COMPILE_CAPABILITY_PROBES=OFF\n")
    set(${output_variable} 0 PARENT_SCOPE)
    return()
  endif()
  set(probe_source "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probes/${probe_name}.cu")
  set(probe_binary "${probe_root}/${probe_name}.cubin")
  separate_arguments(probe_flags NATIVE_COMMAND "${CMAKE_CUDA_FLAGS}")
  set(host_option "")
  if(CMAKE_CUDA_HOST_COMPILER)
    list(APPEND host_option "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
  endif()
  execute_process(
    COMMAND "${CMAKE_CUDA_COMPILER}" ${probe_flags} ${host_option}
            --cubin --std=c++17 -O3 "-arch=sm_${arch}"
            "${probe_source}" -o "${probe_binary}"
    RESULT_VARIABLE probe_result
    OUTPUT_VARIABLE probe_stdout ERROR_VARIABLE probe_stderr
    TIMEOUT 60)
  file(SHA256 "${probe_source}" probe_source_sha256)
  file(WRITE "${probe_root}/${probe_name}.log"
    "target=sm_${arch}\ncompiler=${CMAKE_CUDA_COMPILER}\ncompiler_version=${CMAKE_CUDA_COMPILER_VERSION}\nsource_sha256=${probe_source_sha256}\nresult=${probe_result}\nScope: assembler acceptance; optional isolated-probe SASS evidence is saved separately; neither is runtime evidence.\n${probe_stdout}\n${probe_stderr}\n")
  if(probe_result EQUAL 0 AND EXISTS "${probe_binary}")
    set(${output_variable} 1 PARENT_SCOPE)
    get_filename_component(compiler_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(ARCHBENCH_CUOBJDUMP NAMES cuobjdump HINTS "${compiler_bin}" "${CUDAToolkit_BIN_DIR}")
    find_program(ARCHBENCH_NVDISASM NAMES nvdisasm HINTS "${compiler_bin}" "${CUDAToolkit_BIN_DIR}")
    set(lowering "not_inspected")
    if(ARCHBENCH_CUOBJDUMP AND ARCHBENCH_NVDISASM AND Python3_Interpreter_FOUND)
      get_filename_component(disassembler_bin "${ARCHBENCH_NVDISASM}" DIRECTORY)
      execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH=${disassembler_bin}:$ENV{PATH}"
          "${ARCHBENCH_CUOBJDUMP}" --dump-sass "${probe_binary}"
          RESULT_VARIABLE disassembly_result OUTPUT_FILE "${probe_root}/${probe_name}.sass"
          ERROR_FILE "${probe_root}/${probe_name}.sass.stderr.txt" TIMEOUT 60)
      if(disassembly_result EQUAL 0)
        execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/classify_mma_sass.py"
          --probe "${probe_name}" --sass "${probe_root}/${probe_name}.sass"
          --output "${probe_root}/${probe_name}.codegen.json"
          RESULT_VARIABLE classification_result OUTPUT_VARIABLE classification ERROR_VARIABLE classification_error
          OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 30)
        if(classification_result EQUAL 0 AND classification MATCHES "^(direct_mma_sass|compiler_expanded_mma|scalar_expansion|not_inspected)$")
          set(lowering "${classification}")
        else()
          file(APPEND "${probe_root}/${probe_name}.log" "SASS classification unavailable: ${classification_error}\n")
        endif()
      endif()
    endif()
    set(${output_variable}_CODEGEN "${lowering}" PARENT_SCOPE)
    file(APPEND "${probe_root}/${probe_name}.log" "codegen_evidence=${lowering}\n")
    message(STATUS "Optional ${probe_name} accepted for sm_${arch}; code generation: ${lowering}")
  else()
    set(${output_variable} 0 PARENT_SCOPE)
    message(STATUS "Optional ${probe_name} unavailable for sm_${arch}; software counterpart remains enabled; see ${probe_root}/${probe_name}.log")
  endif()
endfunction()
