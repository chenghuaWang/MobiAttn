add_executable(mobi_attn_arm_flash_attn_2 main.cpp)
target_link_libraries(mobi_attn_arm_flash_attn_2 PRIVATE mafa2 gtest_main)

include(GoogleTest)
