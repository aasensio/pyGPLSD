# pyGPLSD
A Bayesian least-squares deconvolution approach based on Gaussian Processes for the detection of Zeeman signatures in stellar spectropolarimetry.

## Dependencies
The code currently depends on the following packages

	scipy
	matplotlib
	pyDIRECT (https://bitbucket.org/amitibo/pydirect)

The dependency on `pyDIRECT` is for the optimization of the hyperparameters. This package
produces a sufficiently good estimation of the hyperparameters without relying on the 
derivatives of the marginal posterior with respect to the hyperparameters.

## Calling the code
You can see in the code a few examples of how to compute the Bayesian LSD. The code generates
a class that you can use for doing the computations.

	gp = gpLSD(vel, sigma, obs, weights, covarianceFunction=covarianceFunction)

`vel` are the velocity bins, `sigma` is a vector with the noise standard deviation for each spectral
line, `obs` is a matrix containing the value of the Stokes parameter at all velocity bins for 
every spectral line to be considered, `weights` is a vector with the weight for each spectral line,
and `covarianceFunction` defines the possible covariance function (`squared exponential`, `matern32` or `matern52`)

The hyperpameters are optimized by calling

	lambdaGP, sigmaGP = gp.optimizeDirectGP()

and the mean of the final GP prior and its variance is obtained by calling

	gpMean, gpVariance = gp.predict([lambdaGP, sigmaGP])

Finally, the standard least-squares deconvolution profile is obtained by calling

	LSD = gp.computeLSD()