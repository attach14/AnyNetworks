# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension, find_packages
import pybind11
import subprocess
import pdb
import os
import shutil
import sysconfig
import glob

__version__ = "0.1.32"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuildExt(build_ext):
    def run(self):
        for extension in self.extensions:
            self.build_extension(extension)
        return

        
    def build_extension(self, ext):
        build_dir = 'build'
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(['cmake', '-S', '.', '-B', build_dir])
        subprocess.check_call(['cmake', '--build', build_dir])

        so_files = glob.glob(os.path.join(build_dir, '*.so'))
        if not so_files:
            raise RuntimeError("Could not find the built .so file!")
        
        target_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))) + '/hope'
        print(glob.glob(target_dir + '/*'))
        os.makedirs(target_dir, exist_ok=True)

        for so_file in so_files:
            target_path = os.path.join(target_dir, os.path.basename(so_file))
            print(f"Moving {so_file} to {target_path}")
            if os.path.exists(so_file):
                shutil.move(so_file, target_path)
                print("NICE JOB")
            else:
                print(f"File not found: {so_file}")
                raise FileNotFoundError(f"{so_file} does not exist!")


ext_modules = [
    Pybind11Extension(
        "anynetworks",
        ["module_creation.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        language='c++'
    ),
]

setup(
    name='anynetworks',
    version=__version__,
    author='KL',
    description='alpha test',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('_interface')],
    packages=find_packages(include=['hope', 'hope.*']),
    install_requires=[
        'numpy',
        'setuptools',
        'wheel',
        'cmake',
        'pybind11',
        'armadillo'
    ],
    cmdclass={
        'build_ext': CMakeBuildExt,
    },
    zip_safe=False,
    include_package_data=True
)