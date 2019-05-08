all:
	g++ -std=c++11 disparity_filtering.cpp `pkg-config --libs --cflags opencv` -I/usr/local/include -L/usr/local/lib -o teste
r1:
	./teste data/aloeL.png data/aloeR.png s n bm a 15
	./teste data/aloeL.png data/aloeR.png s n sgbm a 5
	./teste data/babyL.png data/babyR.png s n bm b 15
	./teste data/babyL.png data/babyR.png s n sgbm b 5
r2:
	./teste data/aloeL.png data/aloeR.png n s bm a 5
b:
	g++ -std=c++11 pd3.cpp `pkg-config --libs --cflags opencv` -I/usr/local/include -L/usr/local/lib -o bonus
	./bonus
