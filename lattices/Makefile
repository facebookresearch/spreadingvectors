

.SUFFIXES: .cpp .o .cxx


CPPFLAGS=-fPIC -Wall -Wno-sign-compare -g -O3 -mavx -std=c++11 -fopenmp

all: _c_lattices.so

SWIG=swig

c_lattices_wrap.cxx: c_lattices.swig lattice_utils.h lattice_Zn.h
	$(SWIG) -c++ -python c_lattices.swig

c_lattices_wrap.o: c_lattices_wrap.cxx lattice_utils.h lattice_Zn.h

lattice_Zn.o: lattice_utils.h lattice_Zn.h

.cxx.o:
	g++ $(CPPFLAGS) $(EXTRACFLAGS) -c $<

.cpp.o:
	g++ $(CPPFLAGS) $(EXTRACFLAGS) -c $<

c_lattices_wrap.o: EXTRACFLAGS= \
	-I $(shell python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())" ) \
        -I $(shell python -c "import numpy ; print(numpy.get_include())")

# linux-specific link line
_c_lattices.so: c_lattices_wrap.o lattice_utils.o lattice_Zn.o
	g++ -g -shared -fopenmp -o $@ $^

clean:
	rm -f *.o _c_lattices.so c_lattices_wrap.cxx c_lattices.py
