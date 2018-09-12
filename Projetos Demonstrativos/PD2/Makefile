all:
	g++ -std=c++11 calibration.cpp `pkg-config --libs --cflags opencv` -I/usr/local/include -L/usr/local/lib -o calib
	./calib 9 6 8 500 0.5