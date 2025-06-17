set -euo pipefail

pip install .

#等待编译结束

cd tests/cpp/build/
rm -rf *
cmake ..
make 

# 运行 rocprof 并把输出既打印到屏幕又保存到临时文件
ROCLOG=/tmp/rocprof.log
rocprof --stats ./operator/test_operator | tee "$ROCLOG"

# 从 rocprof 输出中提取两组数字 Dimension（2048,12288）
shape_line=$(grep -m 1 'OperatorTest/NormTestSuite.TestNorm/LayerNorm_' "$ROCLOG")
dim1=$(awk -F'X' '{print $3}' <<<"$shape_line")
dim2=$(awk -F'X' '{print $4}' <<<"$shape_line")

# 再提取 ctas_per_row, warps_n, bytes_per_load
ctas=$(grep -m 1 'ctas_per_row:'    "$ROCLOG" | awk -F: '{gsub(/ /,"",$2); print $2}')
wm=$(grep -m 1 'warps_m:'        "$ROCLOG" | awk -F: '{gsub(/ /,"",$2); print $2}')
wn=$(grep -m 1 'warps_n:'        "$ROCLOG" | awk -F: '{gsub(/ /,"",$2); print $2}')
bpl=$(grep -m 1 'bytes_per_load:'   "$ROCLOG" | awk -F: '{gsub(/ /,"",$2); print $2}')

# 拼成文件名并创建空文件
filename="${dim1}_${dim2}_${ctas}_${wm}_${wn}_${bpl}"
touch "/home/tuned_fwd/1024/f16f16/$filename"
echo "→ Created file $filename"

python /home/tools/abs_readall.py "/home/tuned_fwd/1024/f16f16/${filename}"


