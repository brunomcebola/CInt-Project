#!/bin/bash

for file in configs/*; do
  echo "$file"
  python3 proj2.py "$file"
  echo
done
