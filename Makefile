all: vidanalyze faceanalyze

vidanalyze: vidanalyze.o Face.o
	g++  -O2  Face.o vidanalyze.o -o vidanalyze `pkg-config --cflags --libs opencv`
vidanalyze.o:
	g++  -O2 vidanalyze.cpp -c
Face.o:
	g++  -O2  Face.cpp -c
faceanalyze: faceanalyze.o Image.o
	g++  -O2  Image.o faceanalyze.o -o faceanalyze `pkg-config --cflags --libs opencv`
faceanalyze.o:
	g++  -O2  faceanalyze.cpp -c
Image.o:
	g++  -O2  Image.cpp -c


clean:
	rm -f vidanalyze
	rm -f vidanalyze.o
	rm -f Face.o
	rm -f faceanalyze
	rm -f faceanalyze.o
	rm -f Image.o