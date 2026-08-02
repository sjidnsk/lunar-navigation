function(lpp_v3_set_project_options target_name)
  target_compile_features(${target_name} INTERFACE cxx_std_20)

  if(MSVC)
    target_compile_options(
      ${target_name}
      INTERFACE
        /permissive-
        /utf-8
        /Zc:__cplusplus)
    target_compile_definitions(
      ${target_name}
      INTERFACE
        NOMINMAX
        WIN32_LEAN_AND_MEAN)
  endif()
endfunction()
