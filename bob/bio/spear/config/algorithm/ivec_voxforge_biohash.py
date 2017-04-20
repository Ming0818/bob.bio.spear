""" 
Protects IVectors using BioHash in the NORMAL scenario (each user has a unique secret seed, which is used to initialise 
the projection matrix from which the user's BioHash is generated)

Note: The "length" parameter should be changed according to how many bits you wish your BioHash to consist of.  The 
maximum BioHash length should not exceed the value of the "subspace_dimension_of_t" parameter, since the BioHash cannot
have more dimensions than the original IVector.
"""

from bob.bio.gmm.algorithm.IVector_BioHash import IVector_BioHash

algorithm = IVector_BioHash(subspace_dimension_of_t = 50, use_lda = False, use_wccn = False, use_plda = False, length = 10)