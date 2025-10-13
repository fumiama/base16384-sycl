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

# Release 模式性能优化选项
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
if(WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /Oy")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops")
endif()

# Debug 模式选项
if(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()

find_package(IntelSYCL REQUIRED)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")

# Release 模式链接优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(WIN32)
    # Windows 保持原有配置
    set(LINK_FLAGS "${LINK_FLAGS} -flto -fuse-ld=lld -Wl,-O1 -Wl,--as-needed")
  else()
    # Linux 下移除 LTO 相关选项以避免链接问题
    set(LINK_FLAGS "${LINK_FLAGS} -Wl,-O1 -Wl,--as-needed")
  endif()
endif()

add_subdirectory(libs)
message(STATUS "Collected libs: ${B14LIBS}")

if (BUILD STREQUAL "test")
  enable_testing()
  add_subdirectory(tests)
  message(STATUS "Collected tests: ${B14TESTS}")
endif ()

foreach(TARGET_NAME ${B14TESTS})
  target_link_libraries(${TARGET_NAME} ${B14LIBS})
endforeach()
