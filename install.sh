#!/bin/bash

if [ "$1" = true ]; then  
    pip3 install --no-cache-dir /package/.[nn]
else
    pip3 install --no-cache-dir /package/.
fi