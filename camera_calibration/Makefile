all:
	g++ -std=c++11 calibration.cpp `pkg-config --libs --cflags opencv` -I/usr/local/include -L/usr/local/lib -o calib
	./calib 8 6 4 500 0.5