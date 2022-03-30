import numpy as np
#from line_profiler import LineProfiler

class likelihoods():

    '''
    Class with methods for bayesian analysis of different kinds of signals. The methods currently
    include prior and likelihood functions for ISGWB analysis, sky-pixel/radiometer type analysis
    and a power spectra based spherical harmonic analysis.
    '''

    def __init__(self):
        '''
        Init for intializing
        '''
        self.r12 = np.conj(self.r1)*self.r2
        self.r13 = np.conj(self.r1)*self.r3
        self.r21 = np.conj(self.r2)*self.r1
        self.r23 = np.conj(self.r2)*self.r3
        self.r31 = np.conj(self.r3)*self.r1
        self.r32 = np.conj(self.r3)*self.r2
        self.rbar = np.stack((self.r1, self.r2, self.r3), axis=2)

        ## create a data correlation matrix
        self.rmat = np.zeros((self.rbar.shape[0], self.rbar.shape[1], self.rbar.shape[2], self.rbar.shape[2]), dtype='complex')

        for ii in range(self.rbar.shape[0]):
            for jj in range(self.rbar.shape[1]):
                self.rmat[ii, jj, :, :] = np.tensordot(np.conj(self.rbar[ii, jj, :]), self.rbar[ii, jj, :], axes=0 )



    def isgwb_only_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        alpha, log_omega0  = theta

        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        ## change axis order to make taking an inverse easier
        cov_sgwb = Sgw[:, None, None, None]*np.moveaxis(self.response_mat, [-2, -1, -3], [0, 1, 2])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)

        return loglike


    def instr_log_likelihood(self, theta):

        '''
        Calculate likelihood for only instrumental noise


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for  Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        log_Np, log_Na  = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)


        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_noise, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike


    def isgwb_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, alpha, log_omega0 = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike

    def sph_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, alpha, log_omega0  = theta[0],theta[1], theta[2], theta[3]

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata, self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        ## Special case for a truncated power law galactic foreground
        if self.params['modeltype'] == 'dwd_fg' and self.inj['fg_spectrum'] == 'truncated':
            fcutoff = 10**self.inj['log_fcut']
            fcut = (self.fdata < fcutoff)
            Omegaf = (10**log_omega0)*(self.fdata/(self.params['fref']))**alpha
            ## add a negligible amount relative to the true Omegaf to avoid nan errors in log likelihood
            Omegaf = Omegaf*fcut + (self.fdata >= fcutoff)*np.min(Omegaf[Omegaf!=0])*1e-10
        ## WIP for population injections and broken power law
        elif self.params['modeltype'] == 'dwd_fg' and (self.inj['fg_spectrum'] == 'broken_powerlaw' or self.inj['fg_spectrum'] == 'catalogue'):
            ## this may just need to be an entirely separate likelihood tbh
            ## for now let's actually just hardcode the cutoff and second slope...
            fcutoff = 10**self.inj['log_fcut']
            alpha2 = self.inj['alpha2']
            lowfilt = (self.fdata < fcutoff)
            highfilt = np.invert(lowfilt)
            Omega_cut = (10**log_omega0)*(fcutoff/(self.params['fref']))**alpha
            Omegaf = lowfilt*(10**log_omega0)*(self.fdata/(self.params['fref']))**alpha + \
                     highfilt*Omega_cut*(self.fdata/fcutoff)**alpha2
        
        else:
            Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[4:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike
    
    def fg_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis of the Milky Way double white dwarf GW foreground.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''
        
        
        # unpack priors, depends on spectrum model, defaults to standard power law
        if self.params['spectrum_model'] == 'broken_powerlaw':
            log_Np, log_Na, alpha, log_omega0, log_fcutoff, alpha_2  = theta[0],theta[1], theta[2], theta[3], theta[4], theta[5]
        elif self.params['spectrum_model'] == 'truncated_powerlaw':
            log_Np, log_Na, alpha, log_omega0, log_fcutoff  = theta[0],theta[1], theta[2], theta[3], theta[4]
        else:
            log_Np, log_Na, alpha, log_omega0  = theta[0],theta[1], theta[2], theta[3]
            
        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata, self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
#        ## truncated power law galactic foreground
#        if self.params['modeltype'] == 'dwd_fg' and self.params['spectrum_model'] == 'truncated':
#            fcutoff = 10**self.inj['log_fcut']
#            fcut = (self.fdata < fcutoff)
#            Omegaf = (10**log_omega0)*(self.fdata/(self.params['fref']))**alpha
#            ## add a negligible amount relative to the true Omegaf to avoid nan errors in log likelihood
#            Omegaf = Omegaf*fcut + (self.fdata >= fcutoff)*np.min(Omegaf[Omegaf!=0])*1e-10
        ## broken power law model
        if self.params['spectrum_model'] == 'broken_powerlaw':
            lowfilt = (self.fdata < (10**log_fcutoff))
            highfilt = np.invert(lowfilt)
            Omega_cut = (10**log_omega0)*((10**log_fcutoff)/(self.params['fref']))**alpha
            Omegaf = lowfilt*(10**log_omega0)*(self.fdata/(self.params['fref']))**alpha + \
                     highfilt*Omega_cut*(self.fdata/(10**log_fcutoff))**alpha_2
        ## truncated power law model. This is just a broken power law, but alpha2 is fixed (not a parameter)
        ## this is so we can use a very steep second slope (which we can't really measure anyway) to induce a sudden drop in the fg psd.
        elif self.params['spectrum_model'] == 'truncated_powerlaw':
            alpha_2 = self.params['truncation_alpha']
            lowfilt = (self.fdata < (10**log_fcutoff))
            highfilt = np.invert(lowfilt)
            Omega_cut = (10**log_omega0)*((10**log_fcutoff)/(self.params['fref']))**alpha
            Omegaf = lowfilt*(10**log_omega0)*(self.fdata/(self.params['fref']))**alpha + \
                     highfilt*Omega_cut*(self.fdata/(10**log_fcutoff))**alpha_2
        ## defaults to power law
        else:
            Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2
        
        ## broken powerlaw theta has more elements before the blms
        if self.params['spectrum_model'] == 'broken_powerlaw':
            blm_theta = theta[6:]
        elif self.params['spectrum_model'] == 'truncated_powerlaw':
            blm_theta = theta[5:]
        else:
            blm_theta  = theta[4:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike


def bespoke_inv(A):


    """

    compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed

    Credit to Eelco Hoogendoorn at stackexchange for this piece of wizardy. This is > 3 times
    faster than numpy's det and inv methods used in a fully vectorized way as of numpy 1.19.1

    https://stackoverflow.com/questions/21828202/fast-inverse-and-transpose-matrix-in-python

    """


    AI = np.empty_like(A)

    for i in range(3):
        AI[...,i,:] = np.cross(A[...,i-2,:], A[...,i-1,:])

    det = np.einsum('...i,...i->...', AI, A).mean(axis=-1)

    inv_T =  AI / det[...,None,None]

    # inverse by swapping the inverse transpose
    return np.swapaxes(inv_T, -1,-2), det



