function(lpp_v3_set_project_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} INTERFACE /W4 /WX)
  else()
    target_compile_options(
      ${target_name}
      INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Werror)
  endif()
endfunction()
