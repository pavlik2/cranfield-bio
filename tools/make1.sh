export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
clear
rm pr
time g++  cpp_bio_light.cpp -O2 -o pr
./pr
