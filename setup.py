from setuptools import setup, Extension

ext_module = Extension('rir_generator.c_ext',
                       sources=['rir_generator/c_ext/rir_generator_ext.cpp'])

setup(name='rir_generator',
      version='0.1',
      author='mori97',
      author_email='tottexi97131@gmail.com',
      license='GNU General Public License v3.0',
      description='Generating room impulse responses.',
      install_requires=["numpy>=1.18"],
      packages=['rir_generator'],
      ext_modules=[ext_module])