import numpy as np
import numpy.linalg as LA
import jax.numpy as jnp
#import jax.numpy.linalg as JLA
import jax
#from scipy.special import lpmn, sph_harm
from multiprocessing import Pool
from jax import config
config.update("jax_enable_x64", True)
import healpy as hp
from scipy.special import sph_harm
from blip.src.sph_geometry import sph_geometry
from tqdm import tqdm
import os
import time



class fast_geometry(sph_geometry):

    '''
    
    Module containing fast, unified geometry methods. 
    The methods here include calculation of antenna patterns for the Michelson, XYZ, and AET TDI channels.
    The calculations are multilayered and abstracted so as to avoid (almost) all repeated calculations when considering multiple submodels/injection components.
    
    Arguments
    -------------------
    params (dict)  : the parsed parameter dictionary
    nthreads (int) : number of parallel threads to use. Default 1 (non-parallel).
    
    '''

    def __init__(self,params,nthreads=1):
        
        self.params = params
        self.nthreads = nthreads
        if self.nthreads > 1:
            self.parallel = True
        else:
            self.parallel = False
        self.armlength = 2.5e9
        
        ## numpy/jax.numpy switch
        global xp
        backend = jax.default_backend()
        if backend == 'gpu':
            print("GPU detected; performing response function calculations on GPU...")
            self.gpu = True
            xp = jnp
        elif backend == 'cpu':
            print("No GPU detected; performing response function calculations on CPU...")
            self.gpu = False
            xp = np
        else:
            print("Warning: something fishy is afoot! JAX backend is neither CPU nor GPU. Defaulting to CPU; if you are trying to run BLIP on a TPU, don't!")
            self.gpu = False
            xp = np
        
        ## shared memory handling
        if self.gpu and os.environ['TF_FORCE_UNIFIED_MEMORY']:
            self.shared_memory = True


    def lisa_orbits(self, tsegmid):

        '''
        Define LISA orbital positions at the midpoint of each time integration segment using analytic MLDC orbits.

        Parameters
        -----------

        tsegmid  :  array
            A numpy array of the tsegmid for each time integration segment.

        Returns
        -----------
        rs1, rs2, rs3  :  array
            Arrays of satellite positions for each segment midpoint in timearray. e.g. rs1[1] is [x1,y1,z1] at t=midpoint[1]=timearray[1]+(segment length)/2.
        '''
        
        ## Branch orbiting and stationary cases; compute satellite position in stationary case based off of first time entry in data.
        if self.params['lisa_config'] == 'stationary':
            # Calculate start time from tsegmid
            tstart = tsegmid[0] - (tsegmid[1] - tsegmid[0])/2
            # Fill times array with just the start time
            times = np.empty(len(tsegmid))
            times.fill(tstart)
        elif self.params['lisa_config'] == 'orbiting':
            times = tsegmid
        else:
            raise ValueError('Unknown LISA configuration selected')


        ## Semimajor axis in m
        a = 1.496e11


        ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
        betaphase = 0
        alphaphase = 0

        ## Orbital angle alpha(t)
        at = (2*np.pi/31557600)*times + alphaphase

        ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
        e = self.armlength/(2*a*np.sqrt(3))

        ## Initialize arrays
        beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase

        ## meshgrid arrays
        Beta_n, Alpha_t = np.meshgrid(beta_n, at)

        ## Calculate inclination and positions for each satellite.
        x_n = a*np.cos(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.sin(Beta_n) - (1+(np.sin(Alpha_t))**2)*np.cos(Beta_n))
        y_n = a*np.sin(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.cos(Beta_n) - (1+(np.cos(Alpha_t))**2)*np.sin(Beta_n))
        z_n = -np.sqrt(3)*a*e*np.cos(Alpha_t - Beta_n)


        ## Construct position vectors r_n
        rs1 = xp.array([x_n[:, 0],y_n[:, 0],z_n[:, 0]])
        rs2 = xp.array([x_n[:, 1],y_n[:, 1],z_n[:, 1]])
        rs3 = xp.array([x_n[:, 2],y_n[:, 2],z_n[:, 2]])

        return rs1, rs2, rs3

    ###################################################################################
    ## Wrapper functions to return the corresponding response array from its entries ##
    ###################################################################################
    
    def isgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, ghost_arg=None):
        
        '''
        Wrapper function to take sky integral and return response array slice
        '''
        
        R1_ii  = self.dOmega/(4*xp.pi)*xp.sum(F1_ii, axis=1 )
        R2_ii  = self.dOmega/(4*xp.pi)*xp.sum(F2_ii, axis=1 ) 
        R3_ii  = self.dOmega/(4*xp.pi)*xp.sum(F3_ii, axis=1 ) 
        R12_ii = self.dOmega/(4*xp.pi)*xp.sum(F12_ii, axis=1) 
        R13_ii = self.dOmega/(4*xp.pi)*xp.sum(F13_ii, axis=1) 
        R23_ii = self.dOmega/(4*xp.pi)*xp.sum(F23_ii, axis=1) 
        
        return xp.array([ [R1_ii, R12_ii, R13_ii] , [xp.conj(R12_ii), R2_ii, R23_ii], [xp.conj(R13_ii), xp.conj(R23_ii), R3_ii] ])
    
    def sph_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, ghost_arg=None):
        
        '''
        Wrapper function to convolve with Ylms and return response array slice
        '''
        
        R1_ii  = self.dOmega*xp.einsum('ij, jk', F1_ii, self.Ylms)
        R2_ii  = self.dOmega*xp.einsum('ij, jk', F2_ii, self.Ylms)
        R3_ii = self.dOmega*xp.einsum('ij, jk', F3_ii, self.Ylms)
        R12_ii = self.dOmega*xp.einsum('ij, jk', F12_ii, self.Ylms)
        R13_ii = self.dOmega*xp.einsum('ij, jk', F13_ii, self.Ylms)
        R23_ii = self.dOmega*xp.einsum('ij, jk', F23_ii, self.Ylms)
        R21_ii = self.dOmega*xp.einsum('ij, jk', xp.conj(F12_ii), self.Ylms)
        R31_ii = self.dOmega*xp.einsum('ij, jk', xp.conj(F13_ii), self.Ylms)
        R32_ii = self.dOmega*xp.einsum('ij, jk', xp.conj(F23_ii), self.Ylms)
    
        return xp.array([ [R1_ii, R12_ii, R13_ii] , [R21_ii, R2_ii, R23_ii], [R31_ii, R32_ii, R3_ii] ])

    
    def pix_convolved_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii,skymap):
        
        '''
        Wrapper function to convolve with pixel-basis skymap and return response array slice
        
        '''
        
        ## Detector response summed over polarization and integrated over sky direction
        ## The travel time phases for the which are relevent for the cross-channel are
        ## accounted for in the Fplus and Fcross expressions above.
#        R1_ii  = self.dOmega*xp.sum( F1_ii * skymap[None, :], axis=1 )
#        R2_ii  = self.dOmega*xp.sum( F2_ii * skymap[None, :], axis=1 ) 
#        R3_ii  = self.dOmega*xp.sum( F3_ii * skymap[None, :], axis=1 ) 
#        R12_ii = self.dOmega*xp.sum( F12_ii * skymap[None, :], axis=1) 
#        R13_ii = self.dOmega*xp.sum( F13_ii * skymap[None, :], axis=1) 
#        R23_ii = self.dOmega*xp.sum( F23_ii * skymap[None, :], axis=1) 
        R1_ii  = self.dOmega*xp.einsum('ij, j', F1_ii, skymap)
        R2_ii  = self.dOmega*xp.einsum('ij, j', F2_ii, skymap)
        R3_ii  = self.dOmega*xp.einsum('ij, j', F3_ii, skymap)
        R12_ii = self.dOmega*xp.einsum('ij, j', F12_ii, skymap)
        R13_ii = self.dOmega*xp.einsum('ij, j', F13_ii, skymap)
        R23_ii = self.dOmega*xp.einsum('ij, j', F23_ii, skymap)
        
        return xp.array([ [R1_ii, R12_ii, R13_ii] , [xp.conj(R12_ii), R2_ii, R23_ii], [xp.conj(R13_ii), xp.conj(R23_ii), R3_ii] ])
    
    def pix_unconvolved_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, ghost_arg=None):
        '''
        Wrapper function which just returns its inputs, as the unconvolved case doesn't integrate over the sky
        
        '''
        
        return xp.array([ [F1_ii, F12_ii, F13_ii] , [xp.conj(F12_ii), F2_ii, F23_ii], [xp.conj(F13_ii), xp.conj(F23_ii), F3_ii] ])
    
    def pix_masked_unconvolved_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii,mask):
        '''
        Wrapper function which just returns its inputs, as the unconvolved case doesn't integrate over the sky
        
        '''

        return xp.array([ [F1_ii[:,mask], F12_ii[:,mask], F13_ii[:,mask]] , 
                          [xp.conj(F12_ii[:,mask]), F2_ii[:,mask], F23_ii[:,mask]], 
                          [xp.conj(F13_ii[:,mask]), xp.conj(F23_ii[:,mask]), F3_ii[:,mask]] ])
    
    
    ########################################################################
    ## For unpacking per-frequency responses into the correct array shape ##
    ########################################################################
    
    def unpack_wrapper(self,ii,Rf):
        
        '''
        Wrapper function to generically assigen the calculated per-frequency response function slices to their appropriate unique response array.
        
        Arguments
        -------------------------
        ii (int)    : frequency index
        Rf (array)  : Response function slice(s) for each unique response function at frequency index ii
        
        '''
        
        ## generically assign the frequency slice to each unique response
        ## the '...' indexing allows this to handle both 3 x 3 x f x t and 3 x 3 x f x t x n response shapes
        if self.shared_memory:
            for jj in range(len(self.unique_responses)):
                self.unique_responses[jj] = self.unique_responses[jj].at[:,:,ii,...].set(Rf[jj])
        elif self.gpu:
            for jj in range(len(self.unique_responses)):
                self.unique_responses[jj][:,:,ii,...] = np.array(Rf[jj])
        else:
            for jj in range(len(self.unique_responses)):
                self.unique_responses[jj][:,:,ii,...] = Rf[jj]
        
        return
    
    def assign_responses_to_submodels(self):
        
        '''
        Method to attach the unique response functions to their (potentially non-unique) respective submodels.
        
        '''
        
        response_used = np.zeros(len(self.unique_responses),dtype='bool')
        for sm in self.submodels:
            if hasattr(sm,"response_args"):
                rargs = sm.response_args
            else:
                rargs = None
            for jj in range(len(self.unique_responses)):
                if (sm.response_wrapper_func == self.wrappers[jj][0]) and np.array_equal(rargs,self.wrappers[jj][1]):
                    if not self.plot_flag:
                        sm.response_mat = self.unique_responses[jj]
                        ## FIX LATER --- aliasing for now, should standardize everything to sm.response_mat later
                        sm.summ_response_mat = sm.response_mat 
                        if sm.injection:
                            sm.convolve_inj_response_mat()
                    else:
                        sm.unconvolved_fdata_response_mat = self.unique_responses[jj]
                        if sm.injection:
                            sm.convolve_inj_response_mat(fdata_flag=self.plot_flag)
                    response_used[jj] = True
        
        ## safety check to make sure all computed responses were assigned appropriately
        if np.any(np.invert(response_used)):
            print("Warning: responses were calculated but unassigned. This should not happen!")
        
        return
    
    
    ##########################################
    ## For parallelizing across frequencies ##
    ##########################################

    def frequency_response_wrapper(self,ii):
        
        '''
        Wrapper function to help with parallelization of the response function calculations.
        
        Arguments
        --------------
        
        ii (int)   :  Frequency index
        
        Returns
        --------------
        
        response_ii (float array) : Response matrix in that frequency bin
        
        '''
        
        f0_ii = self.f0_jax[ii]
        
        # Calculate GW transfer function for the michelson channels
        gammaU_plus    =    1/2 * (xp.sinc((f0_ii)*(1 - self.udir)/xp.pi)*xp.exp(-1j*f0_ii*(3+self.udir)) + \
                         xp.sinc((f0_ii)*(1 + self.udir)/xp.pi)*xp.exp(-1j*f0_ii*(1+self.udir)))

        gammaV_plus    =    1/2 * (xp.sinc((f0_ii)*(1 - self.vdir)/xp.pi)*xp.exp(-1j*f0_ii*(3+self.vdir)) + \
                         xp.sinc((f0_ii)*(1 + self.vdir)/xp.pi)*xp.exp(-1j*f0_ii*(1+self.vdir)))

        gammaW_plus    =    1/2 * (xp.sinc((f0_ii)*(1 - self.wdir)/xp.pi)*xp.exp(-1j*f0_ii*(3+self.wdir)) + \
                         xp.sinc((f0_ii)*(1 + self.wdir)/xp.pi)*xp.exp(-1j*f0_ii*(1+self.wdir)))


        # Calculate GW transfer function for the michelson channels
        gammaU_minus    =    1/2 * (xp.sinc((f0_ii)*(1 + self.udir)/xp.pi)*xp.exp(-1j*f0_ii*(3 - self.udir)) + \
                         xp.sinc((f0_ii)*(1 - self.udir)/xp.pi)*xp.exp(-1j*f0_ii*(1 - self.udir)))

        gammaV_minus    =    1/2 * (xp.sinc((f0_ii)*(1 + self.vdir)/xp.pi)*xp.exp(-1j*f0_ii*(3 - self.vdir)) + \
                         xp.sinc((f0_ii)*(1 - self.vdir)/xp.pi)*xp.exp(-1j*f0_ii*(1 - self.vdir)))

        gammaW_minus    =    1/2 * (xp.sinc((f0_ii)*(1 + self.wdir)/xp.pi)*xp.exp(-1j*f0_ii*(3 - self.wdir)) + \
                         xp.sinc((f0_ii)*(1 - self.wdir)/xp.pi)*xp.exp(-1j*f0_ii*(1 - self.wdir)))


        ## Michelson antenna patterns
        ## Calculate Fplus
        Fplus1 = 0.5*(self.Fplus_u*gammaU_plus - self.Fplus_v*gammaV_plus)*xp.exp(-1j*f0_ii*(self.udir + self.vdir)/xp.sqrt(3))
        Fplus2 = 0.5*(self.Fplus_w*gammaW_plus - self.Fplus_u*gammaU_minus)*xp.exp(-1j*f0_ii*(-self.udir + self.vdir)/xp.sqrt(3))
        Fplus3 = 0.5*(self.Fplus_v*gammaV_minus - self.Fplus_w*gammaW_minus)*xp.exp(1j*f0_ii*(self.vdir + self.wdir)/xp.sqrt(3))

        ## Calculate Fcross
        Fcross1 = 0.5*(self.Fcross_u*gammaU_plus  - self.Fcross_v*gammaV_plus)*xp.exp(-1j*f0_ii*(self.udir + self.vdir)/xp.sqrt(3))
        Fcross2 = 0.5*(self.Fcross_w*gammaW_plus  - self.Fcross_u*gammaU_minus)*xp.exp(-1j*f0_ii*(-self.udir + self.vdir)/xp.sqrt(3))
        Fcross3 = 0.5*(self.Fcross_v*gammaV_minus - self.Fcross_w*gammaW_minus)*xp.exp(1j*f0_ii*(self.vdir + self.wdir)/xp.sqrt(3))

        ## Detector response averaged over polarization
        ## The travel time phases for the which are relevent for the cross-channel are
        ## accounted for in the Fplus and Fcross expressions above.
        F1_ii  = (1/2)*((xp.absolute(Fplus1))**2 + (xp.absolute(Fcross1))**2)
        F2_ii  = (1/2)*((xp.absolute(Fplus2))**2 + (xp.absolute(Fcross2))**2) 
        F3_ii  = (1/2)*((xp.absolute(Fplus3))**2 + (xp.absolute(Fcross3))**2) 
        F12_ii = (1/2)*(xp.conj(Fplus1)*Fplus2 + xp.conj(Fcross1)*Fcross2)
        F13_ii = (1/2)*(xp.conj(Fplus1)*Fplus3 + xp.conj(Fcross1)*Fcross3)
        F23_ii = (1/2)*(xp.conj(Fplus2)*Fplus3 + xp.conj(Fcross2)*Fcross3)

#        response_slices = [wrapper(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii) if arg is None else wrapper(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, arg) for wrapper, arg in self.wrappers]
        
        response_slices = self.apply_response_slice_convolutions(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii)

        return response_slices
    
    def apply_response_slice_convolutions(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii):
        '''
        Helper function to perform the sky integrals/masking as needed.
        '''
        
        response_slices = []
#        for wrapper, arg in self.wrappers: ## rewrite this to iterate over strings + getattr to avoid pickling functions
#            response_slices.append(wrapper(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, arg))
#        response_slices = [wrapper(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii) if arg is None else wrapper(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, arg) for wrapper, arg in self.wrappers]
            
        
        for wrapper_name, arg in self.wrappers:
            response_slices.append(getattr(self,wrapper_name)(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii, arg))
        
        return response_slices
        
    def get_geometric_Fplus_Fcross(self,arm_hat_ij,mhat_prod,nhat_prod):
        
        '''
        Wrapper function to reduce repeated calculations for the geometric components of the antenna pattern functions
        
        Arguments
        -------------------
        arm_hat_ij (float arrays) : directional unit vector for the arm between satellites i and j.
        mhat_prod, nhat_prod (float arrays) : outer products of phi hat and theta hat, respectively
        
        Returns
        --------------------------
        Fplus_ij, Fcross_ij (float arrays) : Geometric components of the plus and cross antenna pattern functions for LISA arms i and j
        
        '''
        
        interior_prod = xp.einsum("ik,jk -> ijk",arm_hat_ij, arm_hat_ij)
        
        return 0.5*xp.einsum("ijk,ijl", interior_prod, mhat_prod - nhat_prod), 0.5*xp.einsum("ijk,ijl", interior_prod, mhat_prod + nhat_prod)
    
    
    def get_xyz_from_michelson(self):
        
        '''
        Compute the LISA response functions for the XYZ Time-Delay Interferometry channels from the Michelson channels.
        
        '''
        if self.shared_memory:
            for i in range(len(self.unique_responses)):
                if self.unique_responses[i].ndim == 4:
                    self.unique_responses[i] = 4 * self.unique_responses[i] * (xp.sin(2*self.f0[None, None, :, None]))**2
                elif self.unique_responses[i].ndim == 5:
                    for jj in range(self.f0.size):
                        self.unique_responses[i] = self.unique_responses[i].at[:,:,jj,:,:].set(4 * self.unique_responses[i][:,:,jj,:,:] * (xp.sin(2*self.f0[jj]))**2)
    #                self.unique_responses[i] = 4 * R_mich * (np.sin(2*self.f0[None, None, :, None, None]))**2
                else:
                    raise ValueError("Michelson response has an unsupported number of dimensions ({}). Something has gone wrong...".format(self.unique_responses[i].ndim))
        
        else:
            for i in range(len(self.unique_responses)):
                if self.unique_responses[i].ndim == 4:
                    self.unique_responses[i] = 4 * self.unique_responses[i] * (np.sin(2*self.f0[None, None, :, None]))**2
                elif self.unique_responses[i].ndim == 5:
                    for jj in range(self.f0.size):
                        self.unique_responses[i][:,:,jj,:,:] = 4 * self.unique_responses[i][:,:,jj,:,:] * (np.sin(2*self.f0[jj]))**2
    #                self.unique_responses[i] = 4 * R_mich * (np.sin(2*self.f0[None, None, :, None, None]))**2
                else:
                    raise ValueError("Michelson response has an unsupported number of dimensions ({}). Something has gone wrong...".format(self.unique_responses[i].ndim))
            
        return
    
    
    def construct_aet_response_mat(self,xyz_response_mat):
        
        '''
        Calculate the Antenna pattern/detector transfer function for a generic isotropic or anisotropic SGWB using A,E,T TDI channels.

        Parameters
        -----------
        xyz_response_mat (float array) : The XYZ response tensor.
        
        Returns
        ---------
        aet_response_mat (float array) : Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averagedover polarization.
        
        '''

        ## Upnack xyz matrix to make assembling the aet matrix easier
        RXX, RYY, RZZ = xyz_response_mat[0, 0], xyz_response_mat[1, 1], xyz_response_mat[2, 2]
        RXY, RXZ, RYZ = xyz_response_mat[0, 1], xyz_response_mat[0, 2], xyz_response_mat[1, 2]

        ## construct AET matrix elements
        RAA = (1/9) * (4*RXX + RYY + RZZ - 2*RXY - 2*np.conj(RXY) - 2*RXZ - 2*np.conj(RXZ) + \
                        RYZ  + np.conj(RYZ))

        REE = (1/3) * (RZZ + RYY - RYZ - np.conj(RYZ))

        RTT = (1/9) * (RXX + RYY + RZZ + RXY + np.conj(RXY) + RXZ + np.conj(RXZ) + RYZ + np.conj(RYZ))

        RAE = (1/(3*np.sqrt(3))) * (RYY - RZZ - RYZ + np.conj(RYZ) + 2*RXZ - 2*RXY)

        RAT = (1/9) * (2*RXX - RYY - RZZ + 2*RXY - np.conj(RXY) + 2*RXZ - np.conj(RXZ) - RYZ - np.conj(RYZ))

        RET = (1/(3*np.sqrt(3))) * (RZZ - RYY - RYZ + np.conj(RYZ) + np.conj(RXZ) - np.conj(RXY))
        
        if self.shared_memory:
            aet_response_mat = xp.array([ [RAA, RAE, RAT] , \
                                        [xp.conj(RAE), REE, RET], \
                                        [xp.conj(RAT), xp.conj(RET), RTT] ])
        else:
            aet_response_mat = np.array([ [RAA, RAE, RAT] , \
                                        [np.conj(RAE), REE, RET], \
                                        [np.conj(RAT), np.conj(RET), RTT] ])

        return aet_response_mat
    
    def get_aet_from_xyz(self):
        
        '''
        Compute the LISA response functions for the AET Time-Delay Interferometry channels from the XYZ channels.
        
        '''
        
        for i, R_xyz in enumerate(self.unique_responses):
            self.unique_responses[i] = self.construct_aet_response_mat(R_xyz)
        
        return
        
    
    def calculate_response_functions(self,f0,tsegmid,submodels,tdi_lev,plot_flag=False):
        
        '''
        Prototype federated function for response function calculations, designed to eliminate all redundant calculations when computing the response functions for multiple submodels.
        
        
        Parameters
        -----------

        f0 (float array)   : A numpy array of scaled frequencies (f0 = c*f/(2pi*arm_length))

        tsegmid (float array) :  Time axis, as given by the midpoint of every time segment
    
        submodels (models.submodel object) : list of instantiated submodel objects
        
        tdi_lev (str) : The desired level of time-delay interferometry (TDI). Can be 'michelson', 'xyz', or 'aet'.
        
        plot_flag (bool) : If True, the responses will be computed at the data frequencies for plotting purposes. Default False.

        Returns
        -----------
        
        response_mat(s) (numpy array/s) : the appropriate response function for each submodel, attached to that submodel (the function does not return the response arrays directly)
        
        '''
        
        self.submodels = submodels
        self.plot_flag = plot_flag
        ## basic preliminaries
        self.f0 = f0
        self.f0_jax = xp.array(f0)
        
        # Area of each pixel in sq.radians
        self.dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])
        
        ## see if any of the submodels span the full sky
        fullsky = np.any([sm.fullsky for sm in self.submodels if hasattr(sm,"fullsky")])
        
        # Make relevant array of pixel indices
        if fullsky:
            ## if any of the submodels require evaluation over the full sky, we need to do so
            npix = hp.nside2npix(self.params['nside'])
            pix_idx  = np.arange(npix)
        else:
            ## otherwise, make a map of everwhere on the sky where there is power across all submodels
            combined_map = np.sum(np.array([sm.skymap for sm in self.submodels if hasattr(sm,'skymap')]),axis=0)
            
            ## Array of pixel indices where the combined map is nonzero
            pix_idx = np.flatnonzero(combined_map)

        # Angular coordinates of pixel indices
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = np.array([np.sqrt(1-ctheta**2)*np.cos(phi),np.sqrt(1-ctheta**2)*np.sin(phi),ctheta])

        # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
        rs1, rs2, rs3 = self.lisa_orbits(tsegmid)
        
        ## get the unit vectors for each arm, as we need them frequently
        uhat_21 = (rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]
        vhat_31 = (rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]
        what_32 = (rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]
        
        ## Calculate directional unit vector dot products
        ## Dimensions of udir is time-segs x sky-pixels
        self.udir = xp.array(np.einsum('ij,ik',uhat_21,omegahat))
        self.vdir = xp.array(np.einsum('ij,ik',vhat_31,omegahat))
        self.wdir = xp.array(np.einsum('ij,ik',what_32,omegahat))


        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''

        ## we've done some fancy factoring of these calculations to make them as efficient as possible
        
        ######################################################################
        ## mhat and nhat are phi hat and theta hat in cartesian coordinates ##
        ######################################################################
        
        mhat = np.array([np.sin(phi),-np.cos(phi),np.zeros(len(phi))])
        nhat = np.array([np.cos(phi)*ctheta,np.sin(phi)*ctheta,-np.sqrt(1-ctheta**2)])

        ## outer (self-) products of mhat and nhat
        mhat_op = np.einsum("ik,jk -> ijk",mhat,mhat)
        
        nhat_op = np.einsum("ik,jk -> ijk",nhat,nhat)
        
        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        ## we have jitted this function

        self.Fplus_u, self.Fcross_u = self.get_geometric_Fplus_Fcross(uhat_21,mhat_op,nhat_op)
        self.Fplus_v, self.Fcross_v = self.get_geometric_Fplus_Fcross(vhat_31,mhat_op,nhat_op)
        self.Fplus_w, self.Fcross_w = self.get_geometric_Fplus_Fcross(what_32,mhat_op,nhat_op)


        # Set frequency array indices
        idx = range(0,f0.size)
        
        ## step through submodels and determine the appropriate responses, wrappers, etc.
        wrappers = []
        wrapper_args = []
        response_shapes = []
        for sm in self.submodels:
            ## instantiate the response array and specify additional wrappers for unpacking
            ## this is mapped to the response_wrapper_func attached to the submodel
            if sm.spatial_model_name == 'isgwb':
                sm.response_shape = (3,3,f0.size,tsegmid.size)
                sm.response_wrapper_func = "isgwb_wrapper"
                wrappers.append("isgwb_wrapper")
                wrapper_args.append(None)
            elif sm.basis == 'sph':
                ## array size of almax
                alm_size = (sm.almax + 1)**2
                ## initalize array for Ylms
                self.Ylms = np.zeros((npix, alm_size ), dtype='complex')
                ## Get the spherical harmonics
                for ii in range(alm_size):
                    lval, mval = self.idxtoalm(sm.almax, ii)
                    self.Ylms[:, ii] = sph_harm(mval, lval, phi, theta)
                sm.response_shape = (3,3,f0.size, tsegmid.size,alm_size)
                sm.response_wrapper_func = "sph_asgwb_wrapper"
                wrappers.append("sph_asgwb_wrapper")
                wrapper_args.append(None)
            elif sm.basis == 'pixel':
                if sm.injection or hasattr(sm,"fixed_map") and sm.fixed_map:
                    ## templated anisotropic searches with pre-convolved maps
                    sm.response_shape = (3,3,f0.size, tsegmid.size)
                    sm.response_wrapper_func = "pix_convolved_asgwb_wrapper"
                    wrappers.append("pix_convolved_asgwb_wrapper")
                    if fullsky:
                        sm_map = sm.skymap
                    else:
                        sm_map = sm.skymap[pix_idx]
                    ## normalize skymap such that integral of P(n)d2n = 1
                    sm_map = sm_map/(np.sum(sm_map)*self.dOmega)
                    sm.response_args = sm_map
                    wrapper_args.append(sm_map)
                elif hasattr(sm,"mask_idx"):
                    ## unconvolved full-sky pixel response, masked to the desired pixels
                    sm.response_shape = (3,3,f0.size, tsegmid.size, sm.mask_idx.size)
                    ## find a mask that matches the submodel mask to the overall response mask
                    ## if the response is computed over the full sky, this will just be the submodel mask as a boolean array
                    bool_mask = np.array([True if p_idx in sm.mask_idx else False for p_idx in pix_idx])
                    sm.response_args = bool_mask
                    sm.response_wrapper_func = "pix_masked_unconvolved_asgwb_wrapper"
                    wrappers.append("pix_masked_unconvolved_asgwb_wrapper")
                    wrapper_args.append(bool_mask)
                else:
                    ## unconvolved full-sky pixel response
                    sm.response_shape = (3,3,f0.size,tsegmid.size,pix_idx.size)
                    sm.response_wrapper_func = "pix_unconvolved_asgwb_wrapper"
                    wrappers.append("pix_unconvolved_asgwb_wrapper")
                    wrapper_args.append(None)
                    
            else:
                raise ValueError("Specification of the response wrapper is unrecognized. Check the implementation of response_wrapper_func for submodel "+sm.spatial_model_name+" in models.py.")
            
            response_shapes.append(sm.response_shape)

        ## avoid duplicate calculations by reducing the problem to the set of unique responses that need to be computed
        unique_wrappers = []
        unique_shapes = []
        for ii, (wrapper, arg, shape) in enumerate(zip(wrappers,wrapper_args,response_shapes)):
#            if (wrapper,arg) not in unique_wrappers:
#                unique_wrappers.append((wrapper,arg))
#                unique_shapes.append(shape)
            
            ## first one is always unique
            if ii == 0:
                unique_wrappers.append((wrapper,arg))
                unique_shapes.append(shape)
            else:
                ## we can't just as if (wrapper, arg) is in unique_wrappers, because numpy is dumb
                for (previous_wrapper,previous_arg) in unique_wrappers:
                    if not (wrapper == previous_wrapper and np.array_equal(arg,previous_arg)):
                        unique_wrappers.append((wrapper,arg))
                        unique_shapes.append(shape)
                        
                        
        self.wrappers = unique_wrappers
        
        ## if we are using a device with shared RAM (CPU+GPU), we can allow the response arrays to be jax.numpy arrays
        ## otherwise we want to keep the full array on CPU, so use numpy
        if self.shared_memory:
            self.unique_responses = [xp.zeros(shape,dtype='complex128') for shape in unique_shapes]
        else:
            self.unique_responses = [np.zeros(shape,dtype='complex128') for shape in unique_shapes]
        
        
        
        ## Perform the joint computation of the detector response for each frequency
        ## this will, for each frequency, compute the full 3 x 3 x time x frquency_ii x sky direction Michelson responses
        ## and then perform the appropriate sums/convolutions depending on the desired spatial model(s)
        
        ## the parallel implementation still has a lot of overhead due to needing to pickle functions and passing them to the threads
        ## unclear if the effects of this will be noticible at scale
        print("Computing LISA response functions...")
        if self.parallel:
            with Pool(self.nthreads) as pool:
                result = pool.map(self.frequency_response_wrapper,idx)
                for ii, R_f in zip(idx,result):
                    self.unpack_wrapper(ii,R_f)
        ## the non-parallel version has a nice progress bar :)
        else:
            for ii in tqdm(idx):
                R_f = self.frequency_response_wrapper(ii)
                self.unpack_wrapper(ii,R_f)        

        ## handle TDI levels 
        if tdi_lev == 'michelson':
            print("Response functions for Michelson channels complete. Assigning responses...")
        elif tdi_lev == 'xyz':
            print("Assembling XYZ response functions from Michelson responses...")
            self.get_xyz_from_michelson()
            print("Response functions for XYZ channels complete. Assigning responses...")
        elif tdi_lev == 'aet':
            print("Assembling XYZ response from Michelson response...")
            self.get_xyz_from_michelson()
            print("Assembling AET response from XYZ response...")
            self.get_aet_from_xyz()
            print("Response functions for AET channels complete. Assigning responses...")        
        
        ## disseminate the unique responses to their respective (not necessarily unique) submodels
        self.assign_responses_to_submodels()
                
        return

#    def scan_func(self,ii,unique_responses):
#        for jj in range(len(self.unique_responses)):
#            self.unique_responses[jj] = self.unique_responses[jj].at[:,:,ii,...].set(Rf[jj])
#        return self.unpack_wrapper(ii,self.frequency_response_wrapper(ii))
#
#    def scan_func(self,ii): ## list of length number of responses, each ofh which has a frequency axis, but are not necessarily the same shape
#        
#        self.unique_responses, _ = lax.scan(subscan_func,)
#        unique_responses = subscan_func(jj)
#        
#        return 
#    
#    def subscan_func(unique_responses,jj):
#        






def get_model_responses(Model_obj):
    '''
    Wrapper function to call methods for computing the response function.
    
    This exists so that the methods in question can be easily jettisoned when the Model object in JAX JIT compilation.
    
    '''
    ## Having initialized all the components, now compute the LISA response functions
    t1 = time.time()
    fast_rx = fast_geometry(Model_obj.params)
    fast_rx.calculate_response_functions(Model_obj.f0,Model_obj.tsegmid,[Model_obj.submodels[smn] for smn in Model_obj.submodel_names if smn !='noise'],Model_obj.params['tdi_lev'])
    t2 = time.time()
    print("Time elapsed for calculating the LISA response functions for all components via joint computation is {} s.".format(t2-t1))
    ## deallocate to save on memory now that the response functions have been calculated and stored elsewhere
    del fast_rx
    return






























