# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	clang++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` src/simple_ml_ext.cpp -o src/simple_ml_ext.so
