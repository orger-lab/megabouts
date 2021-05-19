# Compute Hankel:
import numpy as np
from scipy.ndimage.interpolation import shift
import scipy 

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def svht(X, sv=None):
    # svht for sigma unknown
    m,n = sorted(X.shape) # ensures m <= n
    beta = m / n # ratio between 0 and 1
    if sv is None:
        sv = scipy.linalg.svdvals(X)
    sv = np.squeeze(sv)
    omega_approx = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return np.median(sv) * omega_approx

class LocallyLinearDynamicalSystem:
    
    def __init__(self,
                 delay):
        
        self.X = None
        self.Y = None
        self.Hankel = None
        self.V = None
        self.delay = delay

    def compute_hankel(self,X):
        H = np.array([])
        H = H.reshape(0,X.shape[0])
        for i in self.delay:
            print(i)
            H = np.concatenate((H,shift(X,(i,0), cval=0).T))#.reshape(NumSeg,tail.shape[0])))
        # Add Intercept:
        H = np.concatenate((H,np.ones_like(H[0,:])[:,np.newaxis].T))#.reshape(NumSeg,tail.shape[0])))
        
        Y = shift(H,(0,-1), cval=0) 
        
        self.X = X
        self.Y = Y
        self.H = H
        return H,Y

    
    def compute_singular_value(self,axis=None):
        # determine rank-reduction
        sv = scipy.linalg.svdvals(self.H)
        tau = svht(self.H, sv=sv)
        r = sum(sv > tau)
        print("rank",r)
        axis.scatter(range(1, len(sv)+1), sv, s=5)
        axis.axhline(tau, c='r')
        axis.set_xlim([0.9,min(len(sv),100)])
        return r,axis

    def compute_singular_vector(self,truncate):
        U2,Sig2,Vh2 = np.linalg.svd(self.H, False) # SVD of input matrix
        r = len(Sig2) if truncate is None else truncate # rank truncation
        self.r = r
        U = U2[:,:r]
        Sig = np.diag(Sig2)[:r,:r]
        V = Vh2.conj().T[:,:r]
        self.U = U
        self.V = V
        self.Sig = Sig
        return U,V,Sig
    
    def reconstruct_X(self,V_):
        H_predicted = self.U@self.Sig@V_
        return H_predicted#[:self.X.shape[0],:]

    def fit_dynamical_system(self,V_):
        V_past = V_[0:-1,:]
        V_future = V_[1:,:]
        all_ones = np.ones(V_past.shape[0])[:,np.newaxis]
        V_past_w_intercept =  np.concatenate((all_ones.T,V_past.T)).T
        res = np.linalg.lstsq(V_past_w_intercept,V_future,rcond=None)
        A_fromV,residual = res[0].T,res[1] # Transpose is crutial to match Atil
        Prediction = A_fromV.dot(V_past_w_intercept.T)
        # Compute Covariance of noise:
        c = A_fromV[:,0]
        A = A_fromV[:,1:]
        noise = V_future - Prediction.T
        covariance_noise = (noise.T.dot(noise))/noise.shape[0]
        theta = (c,A,covariance_noise)
        return theta


    def predict_recursively(self,V0,horizon,theta):
        c = theta[0]
        A = theta[1]
        covariance_noise = theta[2]
        noise_sim = np.random.multivariate_normal(np.zeros(covariance_noise.shape[1]), covariance_noise, size=horizon, check_valid='warn', tol=1e-8)
        V_predicted = np.zeros((horizon,len(V0)))
        V_predicted[0,:] = V0
        for i in range(1,horizon):
            V_predicted[i,:] = c + A@V_predicted[i-1,:] + noise_sim[i,:]
        return V_predicted
    
    #def run_monte_carlo(V0,horizon,c,A,covariance_noise,n):
    def compute_likelihood(self,theta_a,V_b):
                
        V_past = V_b[:,0:-1]
        V_future = V_b[:,1:]

        V_future_pred = np.repeat(theta_a[0][:,np.newaxis],V_past.shape[1],axis=1)+ theta_a[1].dot(V_past) 
        error = V_future-V_future_pred

        det_sigma_a = np.linalg.det(theta_a[2])
        d = len(theta_a[0])
        volume = np.log(np.power(2*np.pi,d)*det_sigma_a)
        precision = np.linalg.inv(theta_a[2])
        mahal = (error.T.dot(precision)*error.T).sum(axis=1)
        likelihood = -1/2 * np.sum(volume-mahal)
        # Careful on the sign (volume + mahal() or (volume - mahal) wikipedia different from article
        return likelihood,volume,mahal

