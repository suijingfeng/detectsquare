# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/suijingfeng/Desktop/detectsquare

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/suijingfeng/Desktop/detectsquare/build

# Include any dependencies generated for this target.
include CMakeFiles/detectsquare.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/detectsquare.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detectsquare.dir/flags.make

CMakeFiles/detectsquare.dir/detectsquare.cpp.o: CMakeFiles/detectsquare.dir/flags.make
CMakeFiles/detectsquare.dir/detectsquare.cpp.o: ../detectsquare.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/suijingfeng/Desktop/detectsquare/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detectsquare.dir/detectsquare.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detectsquare.dir/detectsquare.cpp.o -c /home/suijingfeng/Desktop/detectsquare/detectsquare.cpp

CMakeFiles/detectsquare.dir/detectsquare.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detectsquare.dir/detectsquare.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/suijingfeng/Desktop/detectsquare/detectsquare.cpp > CMakeFiles/detectsquare.dir/detectsquare.cpp.i

CMakeFiles/detectsquare.dir/detectsquare.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detectsquare.dir/detectsquare.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/suijingfeng/Desktop/detectsquare/detectsquare.cpp -o CMakeFiles/detectsquare.dir/detectsquare.cpp.s

CMakeFiles/detectsquare.dir/detectsquare.cpp.o.requires:

.PHONY : CMakeFiles/detectsquare.dir/detectsquare.cpp.o.requires

CMakeFiles/detectsquare.dir/detectsquare.cpp.o.provides: CMakeFiles/detectsquare.dir/detectsquare.cpp.o.requires
	$(MAKE) -f CMakeFiles/detectsquare.dir/build.make CMakeFiles/detectsquare.dir/detectsquare.cpp.o.provides.build
.PHONY : CMakeFiles/detectsquare.dir/detectsquare.cpp.o.provides

CMakeFiles/detectsquare.dir/detectsquare.cpp.o.provides.build: CMakeFiles/detectsquare.dir/detectsquare.cpp.o


# Object files for target detectsquare
detectsquare_OBJECTS = \
"CMakeFiles/detectsquare.dir/detectsquare.cpp.o"

# External object files for target detectsquare
detectsquare_EXTERNAL_OBJECTS =

detectsquare: CMakeFiles/detectsquare.dir/detectsquare.cpp.o
detectsquare: CMakeFiles/detectsquare.dir/build.make
detectsquare: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
detectsquare: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
detectsquare: CMakeFiles/detectsquare.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/suijingfeng/Desktop/detectsquare/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable detectsquare"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detectsquare.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detectsquare.dir/build: detectsquare

.PHONY : CMakeFiles/detectsquare.dir/build

CMakeFiles/detectsquare.dir/requires: CMakeFiles/detectsquare.dir/detectsquare.cpp.o.requires

.PHONY : CMakeFiles/detectsquare.dir/requires

CMakeFiles/detectsquare.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/detectsquare.dir/cmake_clean.cmake
.PHONY : CMakeFiles/detectsquare.dir/clean

CMakeFiles/detectsquare.dir/depend:
	cd /home/suijingfeng/Desktop/detectsquare/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/suijingfeng/Desktop/detectsquare /home/suijingfeng/Desktop/detectsquare /home/suijingfeng/Desktop/detectsquare/build /home/suijingfeng/Desktop/detectsquare/build /home/suijingfeng/Desktop/detectsquare/build/CMakeFiles/detectsquare.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detectsquare.dir/depend

