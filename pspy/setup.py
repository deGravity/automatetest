from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
import zipfile
from sys import platform
from os.path import exists, dirname, realpath, join
import os
import pathlib

def extract_library(lib_name, lib_dir):
    lib_base = join(lib_dir, lib_name)
    if not exists(f'{lib_base}.lib'):
        with zipfile.ZipFile(f'{lib_base}.zip') as zip_ref:
            zip_ref.extractall(lib_dir)

## From https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py ##
class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_%s=%s' % (config.upper(), str(extdir.parent.absolute())),
            '-DCMAKE_BUILD_TYPE=%s' % config
        ]

        # example of build args
        build_args = [
            '--config', config
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

## End from https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py ##

ext_modules = []


if platform == "linux" or platform == "linux2":
    parasolid_library = 'pskernel_archive_linux_x64'
    dn = dirname(realpath(__file__))
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    #extract_library('pskernel_archive_linux_x64', 'parasolid/lib')
    ext_modules = [
        CMakeExtension('pspy_cpp')
    ]
elif platform == "darwin":
    # This is probably computer dependent
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.15"
    parasolid_library = 'pskernel_archive_intel_macos'
    dn = dirname(realpath(__file__))
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    #extract_library('pskernel_archive_linux_x64', 'parasolid/lib')
    ext_modules = [
        CMakeExtension('pspy_cpp')
    ]
elif platform == "win32" or platform == "cygwin":
    parasolid_library = 'pskernel_archive_win_x64'
    dn = dirname(realpath(__file__))
    parasolid_lib_dir = join(dn, 'parasolid', 'lib')
    extract_library(parasolid_library, parasolid_lib_dir)
    ext_modules = [
        CMakeExtension('pspy_cpp')
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
	ext_modules=ext_modules,
	cmdclass={
		'build_ext': build_ext
	},
	packages=find_packages()
)