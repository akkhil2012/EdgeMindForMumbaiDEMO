cmake_minimum_required(VERSION 3.22.1)
project(EdgeMind)

# ── ExecuTorch JNI bridge ──────────────────────────────────────────────────
add_library(executorch_jni SHARED
    ${CMAKE_SOURCE_DIR}/../cpp/executorch_jni.cpp
)

target_include_directories(executorch_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/../cpp/include
)

target_link_libraries(executorch_jni
    android
    log
)

# Link ExecuTorch prebuilt .so if available
find_library(EXECUTORCH_LIB executorch HINTS ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI})
if(EXECUTORCH_LIB)
    target_link_libraries(executorch_jni ${EXECUTORCH_LIB})
endif()

# ── FAISS JNI bridge ──────────────────────────────────────────────────────
add_library(faiss_jni SHARED
    ${CMAKE_SOURCE_DIR}/../cpp/faiss_jni.cpp
)

target_include_directories(faiss_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/../cpp/include
)

target_link_libraries(faiss_jni
    android
    log
)

# Link FAISS prebuilt .so if available
find_library(FAISS_LIB faiss HINTS ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI})
if(FAISS_LIB)
    target_link_libraries(faiss_jni ${FAISS_LIB})
endif()
