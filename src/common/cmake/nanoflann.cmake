FetchContent_Declare(nanoflann
        GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
		GIT_TAG v1.5.0
        )
FetchContent_Populate(nanoflann) # we do not need add_subdirectory() here since we only include the header

add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE ${nanoflann_SOURCE_DIR})
set_target_properties(nanoflann PROPERTIES FOLDER "physsim/extern")