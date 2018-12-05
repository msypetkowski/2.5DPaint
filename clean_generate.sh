#!/bin/bash

declare -a generate_files=(
  "./CMakeCache.txt"
  "./CMakeFiles"
  "./Makefile"
  "./build"
  "./cmake_install.cmake"
  "./2.5DPaint_autogen"
)

for file in "${generate_files[@]}"; do
  if [ -f "$file" ] || [ -d "$file" ] ; then
    echo "Cleaning: ${file}"
    rm -r "$file"
  fi
done

echo "All cmake generate files and project build was cleaned."
