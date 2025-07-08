FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest/
        GIT_TAG 14aa11db02d9851d957f93ef9fddb110c1aafdc6
        )
set(GTEST_INCLUDE_DIR "${googletest_SOURCE_DIR}/googletest/include")
set(BUILD_GMOCK CACHE INTERNAL false)
set(INSTALL_GTEST CACHE INTERNAL true)
set(GTest_DIR CACHE INTERNAL ${GTEST_INCLUDE_DIR})

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set_target_properties(gtest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )
set_target_properties(gtest_main
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )

# Place in a folder
set_target_properties(gtest PROPERTIES FOLDER "extern")
set_target_properties(gtest_main PROPERTIES FOLDER "extern")