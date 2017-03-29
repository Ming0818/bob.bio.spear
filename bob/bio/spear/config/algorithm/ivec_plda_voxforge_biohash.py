from bob.bio.gmm.algorithm.IVector_BioHash import IVector_BioHash

algorithm = IVector_BioHash(
	# IVector parameters
    subspace_dimension_of_t = 100,
    # GMM parameters
    use_lda = True,
    use_wccn = True,
    use_plda = True,
    # BioHash parameters
    length = 50
)