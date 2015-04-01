import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as opt
from DIRECT import solve
import scipy.sparse as sp

try:
	import seaborn as sn
	seabornOK = True
except:
	seabornOK = False

def cholInvert(A):
	"""
	Invert matrix A using a Cholesky decomposition. It works only for symmetric matrices.
	Input
	-----
	A: matrix to invert

	Output
	------
	AInv: matrix inverse
	logDeterminant: returns the logarithm of the determinant of the matrix
	"""
	L = np.linalg.cholesky(A)
	LInv = np.linalg.inv(L)
	AInv = np.dot(LInv.T, LInv)
	logDeterminant = 2.0 * np.sum(np.log(np.diag(L)))
	return AInv, logDeterminant

class gpLSD:
	def __init__(self, velocities, noise, linesObserved, alpha, covarianceFunction='squaredExponential'):
		"""
		This class implements the Bayesian LSD using a Gaussian Process prior

		Input 
		-----
		velocities: vector of size nv with the velocity bins in which all lines are sampled
		noise: array of size nl giving the standard deviation of the noise for each line (nl)
		linesObserved: array of size nv x nl with the observations for each line and velocity bin (nv)
		alpha: array of size nl with the weight associated to each line
		covarianceFunction: selection of the covariance function. We currently have ('squaredExponential', 'matern32', 'matern52')
		                    but more can be trivially added by defining the function below
		"""
		self.epsilon = 1e-8
		self.alpha = alpha
		self.v = velocities		
		self.variance = noise**2
		self.beta = 1.0 / self.variance
		self.nLambda = len(velocities)
		self.nLines = len(alpha)
		self.linesObserved = linesObserved
		self.K = np.zeros((self.nLambda,self.nLambda))
						
		sparseData = []
		sparseRow = []
		sparseCol = []
		for i in range(self.nLambda):
			for j in range(self.nLines):
				sparseRow.append(i*self.nLines+j)
				sparseCol.append(i)
				sparseData.append(alpha[j])
				
		self.W = sp.coo_matrix((sparseData, (sparseRow, sparseCol)), shape=(self.nLines*self.nLambda,self.nLambda))
		
		
		self.AInvDiag = (np.ones(self.nLambda)[:,None] * self.beta).reshape(self.nLines*self.nLambda)		
		print "Computing inverse variance matrix"
		self.AInv = sp.diags(self.AInvDiag, 0)		
		print "Computing determinant of noise matrix"
		self.logA = -np.sum(np.log(self.AInvDiag))	# We use a -1 because we are working with the inverse variance
		print "Precomputing product W^T*V*A^-1*W"		
		self.WTAInvW = self.W.T.dot(self.AInv.dot(self.W)).toarray()
		print "Precomputing product Q = OBS^T * diag(1/sigma^2) * OBS"
		self.Q = self.linesObserved.T.dot(self.AInv.dot(self.linesObserved))		
		print "Precomputing product P=W^T * diag(1/sigma^2) * OBS"		
		self.P = self.W.T.dot(self.AInv.dot(self.linesObserved))		
		print "Precomputing W^T*Obs"
		self.WTAInvObs = self.W.T.dot(self.AInv.dot(self.linesObserved))
		self.AInv = None
		
		if (covarianceFunction == 'squaredExponential'):
			self.covariance = self.squaredExponential
		if (covarianceFunction == 'matern32'):
			self.covariance = self.matern32
		if (covarianceFunction == 'matern52'):
			self.covariance = self.matern52
		
				
	def squaredExponential(self, pars):
		"""
		Squared exponential covariance function
		
		Parameters
		----------
		pars : float
		    array of size 2 with the hyperparameters of the kernel
		
		Returns
		-------
		x : kernel matrix of size nl x nl
		"""
		lambdaGP, sigmaGP = np.exp(pars)
		return sigmaGP * np.exp(-0.5 * lambdaGP * (self.v[:,None]-self.v[None,:])**2)
	
	def matern32(self, pars):
		"""
		Matern nu=3/2 exponential covariance function
		
		Parameters
		----------
		pars : float
		    array of size 2 with the hyperparameters of the kernel
		
		Returns
		-------
		x : kernel matrix of size nl x nl
		"""
		lambdaGP, sigmaGP = np.exp(pars)
		r = np.abs(self.v[:,None]-self.v[None,:])
		y = np.sqrt(3.0) * r / lambdaGP
		return sigmaGP * (1.0 + y) * np.exp(-y)
	
	def matern52(self, pars):
		"""
		Matern nu=5/2 exponential covariance function
		
		Parameters
		----------
		pars : float
		    array of size 2 with the hyperparameters of the kernel
		
		Returns
		-------
		x : kernel matrix of size nl x nl
		"""
		lambdaGP, sigmaGP = np.exp(pars)
		r = np.abs(self.v[:,None]-self.v[None,:])
		y = np.sqrt(5.0) * r / lambdaGP
		return sigmaGP * (1.0 + y + y**2 / 3.0) * np.exp(-y)
	
				
	def marginal(self, pars):
		"""
		Compute the marginal posterior for the hyperparameters
		
		Parameters
		----------
		pars : float array
		    Value of the hyperparameters
		
		Returns
		-------
		logP : float
		    Value of the marginal posterior

		"""
		K = self.covariance(pars)
		
# We apply the Woodbury matrix identity to write the inverse in terms of the inverse of smaller matrices				
		KInv, logK = cholInvert(K + self.epsilon * np.identity(self.nLambda))
		
		t = KInv + self.WTAInvW
		tInv, logt = cholInvert(t)
				
# And use the matrix determinant lemma
		logD = logt + logK + self.logA
		
		logMarginal = -0.5 * self.Q + 0.5 * np.dot(self.P.T, np.dot(tInv, self.P)) - 0.5 * logD		

		return -logMarginal
		
	def _objDirect(self, x, user_data):
		return self.marginal(x), 0
	
	def optimizeDirectGP(self):
		"""
		Optimize the hyperparameters of the GP using the DIRECT optimization method
		
		Returns
		-------
		x: array of float
			Value of the optimal hyperpameters
		"""
		l = [0.05, np.log(np.min(self.variance) / self.nLines)]
		u = [4.0, np.log(10.0*np.max(self.variance) / self.nLines)]
		x, fmin, ierror = solve(self._objDirect, l, u, algmethod=1, maxf=300)
		print 'Optimal lambdaGP={0}, sigmaGP={1} - loglambdaGP={2}, logsigmaGP={3} - logL={4}'.format(np.exp(x[0]), np.exp(x[1]), x[0], x[1], fmin)
		return x[0], x[1]
		
	def predict(self, pars):
		"""
		Prediction using the GP
		
		Parameters
		----------
		pars : array of float
		    value of the hyperpameters
		
		Returns
		-------
		muZ: mean of the GP regression at each velocity bin
		variance: variance of the GP regression at each velocity bin
		"""
		K = self.covariance(pars)
		KInv, _ = cholInvert(K + self.epsilon * np.identity(self.nLambda))
		SigmaInv = KInv + self.WTAInvW
		SigmaZ = np.linalg.inv(SigmaInv)
		muZ = SigmaZ.dot(self.WTAInvObs)
		variance = np.diagonal(SigmaZ)
		return muZ, variance

	def computeLSD(self):
		"""
		Compute the standard LSD deconvolution
		
		Returns
		-------
		x: array of float with the LSD profile
		"""
		linesMatrix = self.linesObserved.reshape((self.nLambda,self.nLines))
		self.ZLSD = np.zeros(self.nLambda)
		for i in range(self.nLambda):
			self.ZLSD[i] = np.sum(self.beta * linesMatrix[i,:] * self.alpha) / np.sum(self.beta * self.alpha**2)
		return self.ZLSD
		
def plotResults(vel, Z, covarianceFunction, outFile=None):
	"""
	Plot the results using the Bayesian LSD and the standard LSD
	
	Parameters
	----------
	vel : float array
	    Velocity bins
	Z : float array
	    exact common signature
	outFile : string
	    name of the output pdf file for the plots. Use None for no output
	covarianceFunction : string
	    covariance function option
	
	Returns
	-------
	None
	"""
	alpha = np.random.rand(nLines)
	fig, ax = pl.subplots(ncols=1, nrows=4, figsize=(8,13), sharex=True)
	ax = ax.flatten()

	noise = sigma * np.random.randn(nLines,nLambda)

	nLinesPartial = [10,50,100,2000]
	for loop in range(len(nLinesPartial)):
	# Generate the observations	
		VObs = alpha[0:nLinesPartial[loop],None] * Z[None,:] + noise[0:nLinesPartial[loop],:]
		VObsVector = np.zeros(nLinesPartial[loop]*nLambda)
		loop2 = 0
		for i in range(nLambda):
			for j in range(nLinesPartial[loop]):
				VObsVector[loop2] = alpha[j] * Z[i] + noise[j,i]
				loop2 += 1
							
		gp = gpLSD(vel, np.ones(nLinesPartial[loop])*sigma, VObsVector, alpha[0:nLinesPartial[loop]], covarianceFunction=covarianceFunction)

		# LSD deconvolution
		ZLSD = gp.computeLSD()

		lambdaGP, sigmaGP = gp.optimizeDirectGP()
		gpMean, gpVariance = gp.predict([lambdaGP, sigmaGP])
		ax[loop].plot(vel,Z)	
		ax[loop].fill_between(vel, gpMean-3.0*np.sqrt(gpVariance), gpMean+3.0*np.sqrt(gpVariance), alpha=0.2)
		ax[loop].fill_between(vel, gpMean-2.0*np.sqrt(gpVariance), gpMean+2.0*np.sqrt(gpVariance), alpha=0.2)
		ax[loop].fill_between(vel, gpMean-np.sqrt(gpVariance), gpMean+np.sqrt(gpVariance), alpha=0.2)	
		ax[loop].plot(vel,ZLSD)
		ax[loop].plot(vel,gpMean)
		ax[loop].set_ylabel('Stokes V')
		ax[loop].text(-3.8,1.7*np.max(Z),'S/N={0:4.1f}'.format(SN*np.sqrt(nLinesPartial[loop])))
		ax[loop].text(-3.8,1.4*np.max(Z),'N$_\mathrm{{lines}}$={0}'.format(nLinesPartial[loop]))
		ax[loop].set_ylim(-2*np.max(Z),2*np.max(Z))
		ax[loop].set_xlim(-4,4)

	ax[-1].set_xlabel('Velocity [km/s]')
	pl.tight_layout()
	if (outFile != None):
		pl.savefig(outFile)

	return

if (__name__ == '__main__'):
	pl.close('all')

	if (seabornOK):
		sn.set_style("dark")

	nLines = 2000
	nLambda = 50
	sigma = 0.5
	ZMax = 0.5
	SN = ZMax / sigma
	vel = np.linspace(-4.0,4.0,nLambda)

	# Product of g*lambda*d
	np.random.seed(123)

	# Example 1
	Z = vel * np.exp(-vel**2)
	Z = ZMax * Z / np.max(Z)
	plotResults(vel, Z, 'matern32', None)

	# Example 2
	Z = vel * np.exp(-vel**2) + 0.2 * vel * np.exp(-(vel-2.5)**2 / 0.5**2)
	Z = ZMax * Z / np.max(Z)
	plotResults(vel, Z, 'matern32', None)