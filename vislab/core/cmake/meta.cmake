FetchContent_Declare(meta
        GIT_REPOSITORY https://github.com/Paul-Hi/meta.git
        GIT_TAG e62f7e43504892ec5422afa24a66052c07ad1520
        )
FetchContent_Populate(meta) # we do not need add_subdirectory() here since we only include the header

add_library(meta INTERFACE)
add_library(meta::meta ALIAS meta)
target_include_directories(meta INTERFACE ${meta_SOURCE_DIR}/src)
