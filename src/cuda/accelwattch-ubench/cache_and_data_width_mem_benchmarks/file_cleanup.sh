#!/bin/bash

# The string to be replaced
OLD_STRING="LDG"

# The new string
NEW_STRING="LDG"

# Use 'find' to get a list of all directories with the old naming convention
# The -depth flag ensures that subdirectories are processed before their parent directories
# This prevents issues where you might rename a parent directory and then fail to find the subdirectories.
find . -depth -type d -name "$OLD_STRING*" | while read dir_path
do
  # Get the new directory path
  new_dir_path=$(echo "$dir_path" | sed "s/$OLD_STRING/$NEW_STRING/")
  
  # Rename the directory
  echo "Renaming directory: $dir_path -> $new_dir_path"
  mv "$dir_path" "$new_dir_path"

  # Change into the new directory to process its contents
  cd "$new_dir_path" || continue

  # Find and replace the string in all files within the directory
  echo "Processing files in: $(pwd)"
  for file in *"$OLD_STRING"*; do
    if [[ -f "$file" ]]; then
      new_file=$(echo "$file" | sed "s/$OLD_STRING/$NEW_STRING/")
      echo "  Renaming file: $file -> $new_file"
      mv "$file" "$new_file"
    fi
  done

  # Find and replace the string in the Makefile
  if [[ -f "Makefile" ]]; then
    echo "  Updating Makefile..."
    sed -i "s/$OLD_STRING/$NEW_STRING/g" Makefile
  fi

  # Change back to the starting directory
  cd - > /dev/null
done

echo "Update complete!"
