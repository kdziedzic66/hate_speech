#!/usr/bin/env bash

function apply_isort {
    local filepath=$1
    echo "applying isort to $filepath"
    isort $filepath
}

function apply_black {
    local filepath=$1
    echo "applying black to $filepath"
    black $filepath
}

if [ $# -eq 0 ] # if this script was executed without any arguments
then
    # get all *.py filepaths returned by git ls-files
    git ls-files | grep .*\\.py | while read -r filepath ; do
        apply_isort $filepath
        apply_black $filepath
    done
else
    for filepath in "$@" # iterate over the input filepaths
    do
        apply_isort $filepath
        apply_black $filepath
    done
fi
