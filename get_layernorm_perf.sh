#!/usr/bin/env bash

find . \
     -name  prof_stats.csv \
     -exec bash -c 'for f; do echo "${f}"; cat "${f}"; done' \
     bash {} + \
| grep --invert-match 'kernel_name,mean_us,std_us' \
| sed 's/prof_stats.csv//g;s/_layernorm_fwd_triton.kd,//g' \
| awk '
/^\.\/[0-9]+_[0-9]+\/$/ {
    gsub(/^\.\//, "", $0)
    gsub(/\/$/, "", $0)
    split($0, nums, "_")
    M = nums[1]
    N = nums[2]
    next
}
/^[0-9.]+,[0-9.]+$/ {
    print M "," N "," $0
}' \
| sed '
s/2048,12288,/1,2048,12288,/;
s/768,1024,/2,768,1024,/;
s/256,65536,/3,256,65536,/;
s/128,6144,/4,128,6144,/;
s/64,2304,/5,64,2304,/;
s/229,541,/6,229,541,/;
s/71,3571,/7,71,3571,/;
s/29,17389,/8,29,17389,/;
s/76800,1600,/9,76800,1600,/' \
| sort -t, -k1,1n
