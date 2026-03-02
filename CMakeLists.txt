# TensorGhost CMakeLists.txt
# Matrix-Free Ghost Penalty Evaluation via Tensor Product Factorization

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# Set the name of the project and target
SET(_main_target TensorGhost)
SET(_main source/dummy.cc)
SET(COMPILE_MAIN FALSE)

# Setup include directories
INCLUDE_DIRECTORIES(include)

# Find deal.II
FIND_PACKAGE(deal.II REQUIRED
  HINTS /scratch/mwichro/dealii/build/spacktbb/ ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(${_main_target})

# Declare all source files the target consists of:
file(GLOB _files source/*cc include/*.h)
file(GLOB _execs execs/*cc)

# Dimensions to build for
SET(DIMENSIONS 2 3)

SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FOREACH(_build_type ${_d2_build_types})
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
    DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
    MESSAGE("-- Found ${_build_type} version of deal.II.")

    STRING(TOUPPER "${_build_type}" _BUILD_TYPE)

    FOREACH(_dim ${DIMENSIONS})
      SET(_p "${${_build_type}_postfix}")
      SET(_lib "${_main_target}${_dim}d${_p}")
      MESSAGE("-- Configuring library ${_lib}")

      ADD_LIBRARY(${_lib} SHARED ${_files})
      TARGET_COMPILE_DEFINITIONS(${_lib} PUBLIC DEAL_DIMENSION=${_dim})
      DEAL_II_SETUP_TARGET(${_lib} ${_BUILD_TYPE})

      # Setup executables
      FOREACH(__exec ${_execs})
        get_filename_component(exec_name ${__exec} NAME_WE)
        SET(_exe "${exec_name}_${_dim}d${_p}")
        MESSAGE("-- Configuring executable ${_exe}")
        ADD_EXECUTABLE(${_exe} ${__exec})
        TARGET_COMPILE_DEFINITIONS(${_exe} PUBLIC DEAL_DIMENSION=${_dim})
        TARGET_LINK_LIBRARIES(${_exe} ${_lib})
        DEAL_II_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
      ENDFOREACH()
    ENDFOREACH()
  ENDIF()
ENDFOREACH()
