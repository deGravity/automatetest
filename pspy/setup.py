from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import zipfile
from sys import platform
from os.path import exists, dirname, realpath, join
import os
import time

def extract_library(lib_name, lib_dir):
    lib_base = join(lib_dir, lib_name)
    if not exists(f'{lib_base}.lib'):
        with zipfile.ZipFile(f'{lib_base}.zip') as zip_ref:
            zip_ref.extractall(lib_dir)

ext_modules = []
install_requires = ['torch']

cpp_sources = [
    'parasolid/parasolid.cpp', 
    'parasolid/frustrum.cpp', 
    'pspy.cpp', 
    'eclass.cpp', 
    'disjointset.cpp', 
    'lsh.cpp',
    'body.cpp',
    'psbody.cpp',
    'psedge.cpp',
    'psface.cpp',
    'psloop.cpp',
    'psvertex.cpp',
    'occtedge.cpp',
    'occtface.cpp',
    'occtloop.cpp',
    'occtvertex.cpp',
    'part.cpp'
]

if platform == "linux" or platform == "linux2":
    parasolid_library = 'pskernel_archive_linux_x64'
    dn = dirname(realpath(__file__))
    eigen3 = join(dn, 'eigen3')
    parasolid = join(dn, 'parasolid')
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    parasolid_lib_file = join(parasolid_lib_dir, f'{parasolid_library}.lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    #extract_library('pskernel_archive_linux_x64', 'parasolid/lib')
    ext_modules = [
        CppExtension(
            'pspy_cpp',
            cpp_sources,
            include_dirs= [eigen3, parasolid],
            extra_objects = [parasolid_lib_file]
        )
    ]
elif platform == "darwin":
    # This is probably computer dependent
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.15"
    parasolid_library = 'pskernel_archive_intel_macos'
    dn = dirname(realpath(__file__))
    eigen3 = join(dn, 'eigen3')
    parasolid = join(dn, 'parasolid')
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    parasolid_lib_file = join(parasolid_lib_dir, f'{parasolid_library}.lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    #extract_library('pskernel_archive_linux_x64', 'parasolid/lib')
    ext_modules = [
        CppExtension(
            # no dot as the relative import gets confused
            'pspy_cpp',
            cpp_sources,
            include_dirs= [eigen3, parasolid],
            extra_objects = [parasolid_lib_file]
        )
    ]
elif platform == "win32" or platform == "cygwin":
    parasolid_library = 'pskernel_archive_win_x64'
    opencascade_tkernel_library = 'TKernel'
    opencascade_tkbrep_library = 'TKBRep'
    opencascade_tkmath_library = 'TKMath'
    opencascade_tkshhealing_library = 'TKShHealing'
    opencascade_tktopalgo_library = 'TKTopAlgo'
    opencascade_tkg3d_library = 'TKG3d'
    dn = dirname(realpath(__file__))
    eigen3 = join(dn, 'eigen3')
    parasolid = join(dn, 'parasolid')
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    opencascade = join(dn, 'opencascade')
    opencascade_lib_dir = join(dn, 'opencascade', 'win64', 'vc14', 'lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    ext_modules = [
        CppExtension(
            'pspy_cpp',
            cpp_sources,
            include_dirs= [eigen3, parasolid, opencascade],
            library_dirs = [parasolid_lib_dir, opencascade_lib_dir],
            libraries = [
                parasolid_library,
                opencascade_tkernel_library,
                opencascade_tkbrep_library,
                opencascade_tkmath_library,
                opencascade_tkshhealing_library,
                opencascade_tktopalgo_library,
                opencascade_tkg3d_library]
        )
    ]

setup(
	name='pspy',
	version='1.0.0',
	author='Ben Jones',
	author_email='benjones@cs.washington.edu',
	url='',
	description='Python wrapper for Parasolid',
	license='MIT',
	python_requires='>=3.6',
	install_requires=install_requires,
	ext_modules=ext_modules,
	cmdclass={
		'build_ext': BuildExtension
	},
	packages=find_packages()
)