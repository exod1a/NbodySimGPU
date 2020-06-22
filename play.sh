#!/bin/bash

start=$(date +%s.%N)
python nbodysim.py
end=$(date +%s.%N)
let dt=$end-$start
echo $dt
