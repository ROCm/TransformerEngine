rm -rf *

cmake ..
make 

rocprof --stats ./operator/test_operator
