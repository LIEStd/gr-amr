INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_AMR amr)

FIND_PATH(
    AMR_INCLUDE_DIRS
    NAMES amr/api.h
    HINTS $ENV{AMR_DIR}/include
        ${PC_AMR_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    AMR_LIBRARIES
    NAMES gnuradio-amr
    HINTS $ENV{AMR_DIR}/lib
        ${PC_AMR_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/amrTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(AMR DEFAULT_MSG AMR_LIBRARIES AMR_INCLUDE_DIRS)
MARK_AS_ADVANCED(AMR_LIBRARIES AMR_INCLUDE_DIRS)
