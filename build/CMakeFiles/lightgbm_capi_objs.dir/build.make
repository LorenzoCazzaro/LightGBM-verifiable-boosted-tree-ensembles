# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lorenzo/Scrivania/LightGBM_Verif_Boosting

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/build

# Include any dependencies generated for this target.
include CMakeFiles/lightgbm_capi_objs.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lightgbm_capi_objs.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lightgbm_capi_objs.dir/flags.make

CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.o: CMakeFiles/lightgbm_capi_objs.dir/flags.make
CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.o: ../src/c_api.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lorenzo/Scrivania/LightGBM_Verif_Boosting/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.o -c /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/src/c_api.cpp

CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/src/c_api.cpp > CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.i

CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/src/c_api.cpp -o CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.s

lightgbm_capi_objs: CMakeFiles/lightgbm_capi_objs.dir/src/c_api.cpp.o
lightgbm_capi_objs: CMakeFiles/lightgbm_capi_objs.dir/build.make

.PHONY : lightgbm_capi_objs

# Rule to build all files generated by this target.
CMakeFiles/lightgbm_capi_objs.dir/build: lightgbm_capi_objs

.PHONY : CMakeFiles/lightgbm_capi_objs.dir/build

CMakeFiles/lightgbm_capi_objs.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lightgbm_capi_objs.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lightgbm_capi_objs.dir/clean

CMakeFiles/lightgbm_capi_objs.dir/depend:
	cd /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lorenzo/Scrivania/LightGBM_Verif_Boosting /home/lorenzo/Scrivania/LightGBM_Verif_Boosting /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/build /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/build /home/lorenzo/Scrivania/LightGBM_Verif_Boosting/build/CMakeFiles/lightgbm_capi_objs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lightgbm_capi_objs.dir/depend

