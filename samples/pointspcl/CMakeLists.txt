enable_language(C)
find_package(PCL 1.3 REQUIRED COMPONENTS visualization)

add_executable(pointspcl pointspcl.cu)
target_link_libraries(pointspcl PRIVATE powermon ${PCL_LIBRARIES})