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
include CMakeFiles/learnCvSeq.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/learnCvSeq.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/learnCvSeq.dir/flags.make

CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o: CMakeFiles/learnCvSeq.dir/flags.make
CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o: ../learnCvSeq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/suijingfeng/Desktop/detectsquare/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o -c /home/suijingfeng/Desktop/detectsquare/learnCvSeq.cpp

CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/suijingfeng/Desktop/detectsquare/learnCvSeq.cpp > CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.i

CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/suijingfeng/Desktop/detectsquare/learnCvSeq.cpp -o CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.s

CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.requires:

.PHONY : CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.requires

CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.provides: CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.requires
	$(MAKE) -f CMakeFiles/learnCvSeq.dir/build.make CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.provides.build
.PHONY : CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.provides

CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.provides.build: CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o


# Object files for target learnCvSeq
learnCvSeq_OBJECTS = \
"CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o"

# External object files for target learnCvSeq
learnCvSeq_EXTERNAL_OBJECTS =

learnCvSeq: CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o
learnCvSeq: CMakeFiles/learnCvSeq.dir/build.make
learnCvSeq: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
learnCvSeq: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
learnCvSeq: CMakeFiles/learnCvSeq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/suijingfeng/Desktop/detectsquare/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable learnCvSeq"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/learnCvSeq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/learnCvSeq.dir/build: learnCvSeq

.PHONY : CMakeFiles/learnCvSeq.dir/build

CMakeFiles/learnCvSeq.dir/requires: CMakeFiles/learnCvSeq.dir/learnCvSeq.cpp.o.requires

.PHONY : CMakeFiles/learnCvSeq.dir/requires

CMakeFiles/learnCvSeq.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/learnCvSeq.dir/cmake_clean.cmake
.PHONY : CMakeFiles/learnCvSeq.dir/clean

CMakeFiles/learnCvSeq.dir/depend:
	cd /home/suijingfeng/Desktop/detectsquare/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/suijingfeng/Desktop/detectsquare /home/suijingfeng/Desktop/detectsquare /home/suijingfeng/Desktop/detectsquare/build /home/suijingfeng/Desktop/detectsquare/build /home/suijingfeng/Desktop/detectsquare/build/CMakeFiles/learnCvSeq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/learnCvSeq.dir/depend

