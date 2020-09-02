"""
SimPack
A variety of classes to analyze molecular dynamics trajectories
"""

# Add imports here
#from . import simpack

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
