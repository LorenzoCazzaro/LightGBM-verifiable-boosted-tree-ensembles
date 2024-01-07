file(REMOVE_RECURSE
  "../lib_lightgbm.pdb"
  "../lib_lightgbm.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/_lightgbm.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
