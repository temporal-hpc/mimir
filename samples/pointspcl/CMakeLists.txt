#enable_language(C)
#find_package(PCL 1.3 REQUIRED COMPONENTS visualization)

include(FetchContent)
FetchContent_Declare(pcl
    GIT_REPOSITORY https://github.com/PointCloudLibrary/pcl.git
    GIT_TAG        605cdd5b1b971d3e90c737aaae947bbba41b66e7 # pcl-1.15.0
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3147391d946bb4b6c68edd901f2add6ac1f31f8c # 3.4.0
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(eigen pcl)

add_executable(pointspcl pointspcl.cu)
target_link_libraries(pointspcl PRIVATE powermon pcl) #${PCL_LIBRARIES})