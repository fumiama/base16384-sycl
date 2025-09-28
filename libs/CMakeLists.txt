file(GLOB CPP_FILES "*.cpp")

set(LOCAL_B14LIBS "")

foreach(CPP_FILE ${CPP_FILES})
  # name without .cpp
  get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)
  message(STATUS "Add lib: ${TARGET_NAME}")
  add_library(${TARGET_NAME} STATIC ${CPP_FILE})

  set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
  
  list(APPEND LOCAL_B14LIBS ${TARGET_NAME})
endforeach()

set(B14LIBS ${LOCAL_B14LIBS} PARENT_SCOPE)
