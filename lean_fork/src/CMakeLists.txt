cmake_minimum_required(VERSION 3.16.3)
project(LEAN CXX C)

add_executable(ml_server main.cpp ml_server.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../lean/src/cmake/Modules")
find_package(GMP 5.0.5 REQUIRED)
include_directories(${GMP_INCLUDE_DIR})


target_include_directories(ml_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../lean/src)
target_include_directories(ml_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../checkpoint-process)


set(LIBLEAN_PATH "release")
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBLEAN_PATH "debug")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)
target_link_libraries(ml_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../lean/build/${LIBLEAN_PATH}/libleanstatic.a)  # should be debug if debug, release if release ?

# target_include_directories(ml_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../checkpoint-process)
# LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../checkpoint-process/build/${LIBLEAN_PATH})

add_library( checkpoint SHARED IMPORTED GLOBAL)
set_target_properties( checkpoint PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../checkpoint-process/build/${LIBLEAN_PATH}/libcheckpoint.so )

set(EXTRA_LIBS ${GMP_LIBRARIES} pthread dl checkpoint)

target_link_libraries(ml_server PRIVATE ${EXTRA_LIBS})

set(LEAN_EXTRA_CXX_FLAGS "${LEAN_EXTRA_CXX_FLAGS} -D LEAN_JSON")
set(LEAN_EXTRA_CXX_FLAGS "${LEAN_EXTRA_CXX_FLAGS} -D LEAN_MULTI_THREAD")

set(CMAKE_CXX_FLAGS                "${LEAN_EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS                "-Wall -Wextra -std=c++17 -g3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG          "-g3 ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3 ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

install(TARGETS ml_server DESTINATION bin)