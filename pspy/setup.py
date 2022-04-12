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
    'occtbody.cpp',
    'occtedge.cpp',
    'occtface.cpp',
    'occtloop.cpp',
    'occtvertex.cpp',
    'part.cpp'
    'edge.cpp',
    'face.cpp',
    'loop.cpp',
    'vertex.cpp',
    'part.cpp',
    'implicit_part.cpp'
]

if platform == "linux" or platform == "linux2":
    parasolid_library = 'pskernel_archive_linux_x64'
    opencascade_tkernel_library = 'TKernel'
    opencascade_tkxsbase_library = 'TKXSBase'
    opencascade_tkstep_library = 'TKSTEP'
    opencascade_tkbrep_library = 'TKBRep'
    opencascade_tkg3d_library = 'TKG3d'
    opencascade_tkmath_library = 'TKMath'
    opencascade_tkmesh_library = 'TKMesh'
    opencascade_tktopalgo_library = 'TKTopAlgo'
    opencascade_tkshhealing_library = 'TKShHealing'
    dn = dirname(realpath(__file__))
    inc_dir = os.getenv('LIBRARY_INC')
    lib_dir = os.getenv('LIBRARY_LIB')
    eigen3 = join(inc_dir, 'eigen3')
    parasolid = join(dn, 'parasolid')
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    parasolid_lib_file = join(parasolid_lib_dir, f'{parasolid_library}.lib')
    opencascade = join(inc_dir, 'opencascade')
    opencascade_tkernel_lib_file = join(lib_dir, f'{opencascade_tkernel_library}.lib')
    opencascade_tkxsbase_lib_file = join(lib_dir, f'{opencascade_tkxsbase_library}.lib')
    opencascade_tkstep_lib_file = join(lib_dir, f'{opencascade_tkstep_library}.lib')
    opencascade_tkbrep_lib_file = join(lib_dir, f'{opencascade_tkbrep_library}.lib')
    opencascade_tkg3d_lib_file = join(lib_dir, f'{opencascade_tkg3d_library}.lib')
    opencascade_tkmath_lib_file = join(lib_dir, f'{opencascade_tkmath_library}.lib')
    opencascade_tkmesh_lib_file = join(lib_dir, f'{opencascade_tkmesh_library}.lib')
    opencascade_tktopalgo_lib_file = join(lib_dir, f'{opencascade_tktopalgo_library}.lib')
    opencascade_tkshhealing_lib_file = join(lib_dir, f'{opencascade_tkshhealing_library}.lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    #extract_library('pskernel_archive_linux_x64', 'parasolid/lib')
    ext_modules = [
        CppExtension(
            'pspy_cpp',
            cpp_sources,
            include_dirs= [eigen3, parasolid, opencascade],
            extra_objects = [
                parasolid_lib_file,
                opencascade_tkernel_lib_file,
                opencascade_tkbrep_lib_file,
                opencascade_tkmath_lib_file,
                opencascade_tkshhealing_lib_file,
                opencascade_tktopalgo_lib_file,
                opencascade_tkg3d_lib_file,
                opencascade_tkstep_lib_file,
                opencascade_tkmesh_lib_file,
                opencascade_tkxsbase_lib_file]
        )
    ]
elif platform == "darwin":
    # This is probably computer dependent
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.15"
    parasolid_library = 'pskernel_archive_intel_macos'
    opencascade_tkernel_library = 'TKernel'
    opencascade_tkxsbase_library = 'TKXSBase'
    opencascade_tkstep_library = 'TKSTEP'
    opencascade_tkbrep_library = 'TKBRep'
    opencascade_tkg3d_library = 'TKG3d'
    opencascade_tkmath_library = 'TKMath'
    opencascade_tkmesh_library = 'TKMesh'
    opencascade_tktopalgo_library = 'TKTopAlgo'
    opencascade_tkshhealing_library = 'TKShHealing'
    dn = dirname(realpath(__file__))
    inc_dir = os.getenv('LIBRARY_INC')
    lib_dir = os.getenv('LIBRARY_LIB')
    eigen3 = join(inc_dir, 'eigen3')
    parasolid = join(dn, 'parasolid')
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    parasolid_lib_file = join(parasolid_lib_dir, f'{parasolid_library}.lib')
    opencascade = join(inc_dir, 'opencascade')
    opencascade_tkernel_lib_file = join(lib_dir, f'{opencascade_tkernel_library}.lib')
    opencascade_tkxsbase_lib_file = join(lib_dir, f'{opencascade_tkxsbase_library}.lib')
    opencascade_tkstep_lib_file = join(lib_dir, f'{opencascade_tkstep_library}.lib')
    opencascade_tkbrep_lib_file = join(lib_dir, f'{opencascade_tkbrep_library}.lib')
    opencascade_tkg3d_lib_file = join(lib_dir, f'{opencascade_tkg3d_library}.lib')
    opencascade_tkmath_lib_file = join(lib_dir, f'{opencascade_tkmath_library}.lib')
    opencascade_tkmesh_lib_file = join(lib_dir, f'{opencascade_tkmesh_library}.lib')
    opencascade_tktopalgo_lib_file = join(lib_dir, f'{opencascade_tktopalgo_library}.lib')
    opencascade_tkshhealing_lib_file = join(lib_dir, f'{opencascade_tkshhealing_library}.lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    #extract_library('pskernel_archive_linux_x64', 'parasolid/lib')
    ext_modules = [
        CppExtension(
            # no dot as the relative import gets confused
            'pspy_cpp',
            cpp_sources,
            include_dirs= [eigen3, parasolid, opencascade],
            extra_objects = [
                parasolid_lib_file,
                opencascade_tkernel_lib_file,
                opencascade_tkbrep_lib_file,
                opencascade_tkmath_lib_file,
                opencascade_tkshhealing_lib_file,
                opencascade_tktopalgo_lib_file,
                opencascade_tkg3d_lib_file,
                opencascade_tkstep_lib_file,
                opencascade_tkmesh_lib_file,
                opencascade_tkxsbase_lib_file]
        )
    ]
elif platform == "win32" or platform == "cygwin":
    parasolid_library = 'pskernel_archive_win_x64'
    opencascade_tkernel_library = 'TKernel'
    opencascade_tkxsbase_library = 'TKXSBase'
    opencascade_tkstep_library = 'TKSTEP'
    opencascade_tkbrep_library = 'TKBRep'
    opencascade_tkg3d_library = 'TKG3d'
    opencascade_tkmath_library = 'TKMath'
    opencascade_tkmesh_library = 'TKMesh'
    opencascade_tktopalgo_library = 'TKTopAlgo'
    opencascade_tkshhealing_library = 'TKShHealing'
    dn = dirname(realpath(__file__))
    inc_dir = os.getenv('LIBRARY_INC')
    lib_dir = os.getenv('LIBRARY_LIB')
    eigen3 = join(inc_dir, 'eigen3')
    parasolid = join(dn, 'parasolid')
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    opencascade = join(inc_dir, 'opencascade')
    extract_library(parasolid_library, parasolid_lib_dir)
    ext_modules = [
        CppExtension(
            'pspy_cpp',
            cpp_sources,
            include_dirs= [eigen3, parasolid, opencascade],
            library_dirs = [parasolid_lib_dir, lib_dir],
            libraries = [
                parasolid_library,
                opencascade_tkernel_library,
                opencascade_tkbrep_library,
                opencascade_tkmath_library,
                opencascade_tkshhealing_library,
                opencascade_tktopalgo_library,
                opencascade_tkg3d_library,
                opencascade_tkstep_library,
                opencascade_tkmesh_library,
                opencascade_tkxsbase_library]
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