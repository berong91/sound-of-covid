#!/bin/bash

if [ -z "$(ls -A Coswara_Data)" ]; then
   git submodule update --init --recursive
fi

git submodule update --recursive --remote

find Coswara_Data -mindepth 1 -maxdepth 1 -not -path '*/\.*' -not -path . -type d -exec sh -c '(cd {}; pwd; cat *.tar.gz.* | tar -k -C "../../extracted" -zvxf -)' \;

rsync -avP --exclude='.*' --include='*.csv' --include='*/' --exclude='*' ./Coswara_Data/ ./extracted
