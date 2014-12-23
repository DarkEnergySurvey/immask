import distutils
from distutils.core import setup

# The main call
setup(name='immask',
      version ='0.2.3',
      license = "GPL",
      description = "The Image Masking Module for DESDM.",
      author = "Felipe Menanteau",
      author_email = "felipe@illinois.edu",
      packages = ['immask'],
      package_dir = {'': 'python'},
      scripts = ['bin/immask',
                 'bin/example_immask.py',
                 'bin/source_me_lsst'],
      )





