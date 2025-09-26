# Please run setvars.sh in your oneAPI root dir first before CMake.
# In Windows, you need to run setvars.bat and CMake in CMD (not PS).

if(UNIX)
  # Direct CMake to use icpx rather than the default C++ compiler/linker
  set(CMAKE_C_COMPILER icx)
  set(CMAKE_CXX_COMPILER icpx)
  # Also set the linker to use icpx
  set(CMAKE_CXX_LINKER icpx)
  set(CMAKE_LINKER icpx)
endif()

if(WIN32)
  include (Platform/Windows-Clang)
  # This is a Windows-specific flag that enables exception handling in host code
  set(WIN_FLAG "/EHsc")
  # Set CMake to use icx-cl rather than the default C++ compiler/linker
  set(CMAKE_C_COMPILER icx-cl)
  set(CMAKE_CXX_COMPILER icx-cl)
  # Also set the linker to use icpx
  set(CMAKE_CXX_LINKER icx-cl)
  set(CMAKE_LINKER icx-cl)
endif()

cmake_minimum_required(VERSION 3.4...4.1.1)
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(
  base16384
  VERSION 2.3.2
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /execution-charset:utf-8 /source-charset:utf-8")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finput-charset=utf-8 -fexec-charset=utf-8")
endif()

find_package(IntelSYCL REQUIRED)

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")

enable_testing()
add_subdirectory(tests)
