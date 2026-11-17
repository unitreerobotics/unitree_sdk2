# Recursively walk example/ subdirectories and install every add_executable()
# target to <prefix>/bin. New examples do not need to repeat install(TARGETS ...)
# in each CMakeLists.txt, and the installed bin/ matches the build tree's bin/.
function(_unitree_sdk2_install_example_targets dir)
    get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(_sub ${_subdirs})
        _unitree_sdk2_install_example_targets("${_sub}")
    endforeach()

    get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target ${_targets})
        get_target_property(_type "${_target}" TYPE)
        if(_type STREQUAL "EXECUTABLE")
            install(TARGETS "${_target}" RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
        endif()
    endforeach()
endfunction()

_unitree_sdk2_install_example_targets("${CMAKE_CURRENT_SOURCE_DIR}/example")
