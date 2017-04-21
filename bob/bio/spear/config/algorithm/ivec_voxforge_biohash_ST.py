""" 
Protects IVectors using BioHash in the STOLEN TOKEN scenario (impostor steals genuine user's secret seed, so projection 
matrix used to generate the impostor's BioHash is the same as that used to generate the genuine user's BioHash)

Notes: 

1. The "length" parameter should be changed according to how many bits you wish your BioHash to consist of.  The 
maximum BioHash length should not exceed the value of the "subspace_dimension_of_t" parameter, since the BioHash cannot
have more dimensions than the original IVector.

2. The "user_seed" parameter can be set to any integer.  In this configuration file, user_seed = length for simplicity
and ease of debugging.
"""

from bob.bio.gmm.algorithm.IVector_BioHash import IVector_BioHash

algorithm = IVector_BioHash(subspace_dimension_of_t = 400, use_lda = False, use_wccn = False, use_plda = False, length = 10, user_seed = 10)