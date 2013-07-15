# Python setup

from distutils.core import setup, Extension
from numpy import get_include

numpy_inc = get_include()		#  NumPy include path.
shtns_o = ['SHT.o', 'sht_std.o', 'sht_ltr.o', 'sht_m0.o', 'sht_m0ltr.o']
libdir = "/home/mike"
if len(libdir) == 0:
	libdir = []
else:
	libdir = [libdir+"/lib"]
cargs = "-fopenmp"
libs = "-lfftw3_omp -lfftw3 -lrt -lm "
libslist = libs.replace('-l','').split()	# transform to list of libraries

shtns_module = Extension('_shtns', sources=['shtns_numpy_wrap.c'],
	extra_objects=shtns_o, depends=shtns_o,
	extra_compile_args=cargs.split(),
	library_dirs=libdir,
	libraries=libslist,
	include_dirs=[numpy_inc])

setup(name='SHTns',
	version='2.3.1',
	description='High performance Spherical Harmonic Transform',
	license='CeCILL',
	author='Nathanael Schaeffer',
	author_email='nschaeff@ujf-grenoble.fr',
	url='https://bitbucket.org/nschaeff/shtns',
	ext_modules=[shtns_module],
	py_modules=["shtns"],
	)
