file(GLOB CPP_FILES "*.cpp")

foreach(CPP_FILE ${CPP_FILES})
  # name without .cpp
  get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)
  message(STATUS "Add test ${TARGET_NAME}")
  add_executable(${TARGET_NAME} ${CPP_FILE})

  set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

  add_test(NAME test_${TARGET_NAME} COMMAND ${TARGET_NAME})
endforeach()
