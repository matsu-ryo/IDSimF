project(RFQDSim)

set(SOURCE_FILES
        RFQDSim.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} core apputils particlesimulation file_io integration spacecharge collisionmodels rs)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})

# register an ideal RFQD smoke test for CTest
add_test(NAME app_rfqd_RFQDSim_ideal COMMAND ${PROJECT_NAME}
        "example/idealRFQD.json" "run_app_rfqd_RFQDSim_ideal" -n ${N_THREADS})

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/example/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/example/)
