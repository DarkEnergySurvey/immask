import distutils
from distutils.core import setup

# The main call
setup(name='immask',
      version ='3.0.0',
      license = "GPL",
      description = "The Image Masking Module for DESDM.",
      author = "Felipe Menanteau, Alex Drlica-Wagner, Eli Rykoff",
      author_email = "felipe@illinois.edu, kadrlica@fnal.gov, erykoff@slac.stanford.edu",
      packages = ['immask'],
      package_dir = {'': 'python'},
      scripts = ['bin/immask',
                 'bin/example_immask.py',
                 'bin/source_me_lsst'],
      )
