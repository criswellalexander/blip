import numpy as np
import jax
import jax.numpy as jnp
import os
from tqdm import tqdm
from jax import config
config.update("jax_enable_x64", True)

class LISAdata():

    '''
    Class for lisa data. Includes methods for generation of gaussian instrumental noise, and generation
    of isotropic stochastic background. Any eventually signal models should be added as methods here. This
    has the Antennapatterns class as a super class.
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
        
        ## numpy/jax.numpy switch
        global xp
        backend = jax.default_backend()
        if backend == 'gpu':
            print("GPU detected; performing data simulation on GPU...")
            self.gpu = True
            xp = jnp
        elif backend == 'cpu':
            print("No GPU detected; performing data simulation on CPU...")
            self.gpu = False
            xp = np
        else:
            print("Warning: something fishy is afoot! JAX backend is neither CPU nor GPU. Defaulting to CPU; if you are trying to run BLIP on a TPU, don't!")
            self.gpu = False
            xp = np
            
    ## Method for reading frequency domain spectral data if given in an npz file
    def read_spectrum(self):

        '''
        Read an input frequency domain data file. Returns the fourier transform of the data from the three channels and an array of reference frequencyes

        Returns
        ---------

        rA, rE, rT, fdata   :   float

        '''

        if os.path.isfile(self.params['input_spectrum']) and not self.params['doPreProc']:
            print("loading freq domain data from input file")

            data = np.load(self.params['out_dir'] + '/' +self.params['input_spectrum'])
            r1    = data['r1']
            r2    = data['r2']
            r3    = data['r3']
            fdata = data['fdata']

            return r1, r2, r3, fdata



    def add_sgwb_data(self, injmodel, key, tbreak = 0.0):
        
        '''
        Function to simulate the SGWB time series, given a spectrum.
        
        Arguments
        --------------------
        injmodel (Injection component) : The injection component submodel.
        key (int) : Key for the numpy (or JAX) rng
        
        '''
 
        N = self.Injection.Npersplice
        halfN = int(0.5*N)
        
        ## compute the astrophysical spectrum
        injmodel_args = [injmodel.truevals[parameter] for parameter in injmodel.spectral_parameters]
        
        Sgw = injmodel.compute_Sgw(self.Injection.frange,injmodel_args)
        
        injmodel.frozen_spectra = Sgw
        
        ## the spectrum of the frequecy domain gaussian for ifft
        norms = xp.sqrt(self.params['fs']*Sgw*N)/2

        ## index array for one segment
        t_arr = xp.arange(N)

        ## the window for splicing
        splice_win = xp.sin(xp.pi * t_arr/N)
            
        ## create rng, etc.
        if self.gpu:
            jax_key = jax.random.key(key)
        else:
            numpy_rng = np.random.default_rng(key)
        
        ## Loop over splice segments
        print("Simulating time-domain data for component '{}'...".format(injmodel.name))
        for ii in tqdm(range(self.Injection.nsplice)):
            ## move frequency to be the zeroth-axis, then cholesky decomp
            ## this sometimes breaks on GPU for unclear reasons, hence the try/except
            try:
                L_cholesky = norms[:, None, None] *  xp.linalg.cholesky(xp.moveaxis(injmodel.inj_response_mat[:, :, :, ii], -1, 0))
            except:
                L_cholesky = norms[:, None, None] *  xp.array(np.linalg.cholesky(np.moveaxis(injmodel.inj_response_mat[:, :, :, ii], -1, 0)))
            
            ## generate standard normal complex data first
            if self.gpu:
                _, jax_key, jax_key_2 = jax.random.split(jax_key,3) ## needed to actually produce a new set of random numbers every time through the loop!
                z_norm = jax.random.normal(jax_key,shape=(self.Injection.frange.size, 3)) + 1j * jax.random.normal(jax_key_2,shape=(self.Injection.frange.size, 3))  
            else:
                z_norm = numpy_rng.normal(size=(self.Injection.frange.size, 3)) + 1j * numpy_rng.normal(size=(self.Injection.frange.size, 3))

            ## The data in z_norm is rescaled into z_scale using L_cholesky
            z_scale = xp.einsum('ijk, ikl -> ijl', L_cholesky, z_norm[:, :, None])[:, :, 0]

            ## The three channels : concatenate with norm at f = 0 to be zero
            zero_arr = xp.zeros(1)
            htilda1  = xp.concatenate([ zero_arr, z_scale[:, 0]])
            htilda2  = xp.concatenate([ zero_arr, z_scale[:, 1]])
            htilda3  = xp.concatenate([ zero_arr, z_scale[:, 2]])


            if ii == 0:
                # Take inverse fft to get time series data
                h1 = splice_win * xp.fft.irfft(htilda1, N)
                h2 = splice_win * xp.fft.irfft(htilda2, N)
                h3 = splice_win * xp.fft.irfft(htilda3, N)

            else:

                ## First append half-splice worth of zeros
                h1 = xp.append(h1, xp.zeros(halfN))
                h2 = xp.append(h2, xp.zeros(halfN))
                h3 = xp.append(h3, xp.zeros(halfN))

                ## Then add the new splice segment
                if self.gpu:
                    h1 = h1.at[-N:].set(h1[-N:] + splice_win * xp.fft.irfft(htilda1, N))
                    h2  = h2.at[-N:].set(h2[-N:] + splice_win * xp.fft.irfft(htilda2, N))
                    h3 = h3.at[-N:].set(h3[-N:] + splice_win * xp.fft.irfft(htilda3, N))
                else:
                    h1[-N:] = h1[-N:] + splice_win * xp.fft.irfft(htilda1, N)
                    h2[-N:] = h2[-N:] + splice_win * xp.fft.irfft(htilda2, N)
                    h3[-N:] = h3[-N:] + splice_win * xp.fft.irfft(htilda3, N)


        ## remove the first half and the last half splice.
        h1, h2, h3 = h1[halfN:-halfN], h2[halfN:-halfN], h3[halfN:-halfN]

        tarr = self.params['tstart'] + tbreak +  xp.arange(0, self.params['dur'], 1.0/self.params['fs'])

        return h1, h2, h3, tarr

    def read_data(self):

        '''
        Read mldc or other external domain data from an ascii txt file. Since this was used primarily for
        the MLDC, it assumes that the data is given in X,Y and Z channels.
        Returns
        ---------

        h1, h2, h3 : float
            Time series data for the three TDI channels


        '''

        hoft = np.loadtxt(self.params['datafile'])

        fs_default = 1.0/(hoft[1, 0] - hoft[0, 0])

        ## Read in the duration seconds of data + one segment of buffer
        end_idx = int((self.params['dur'] + self.params['seglen'])*fs_default)

        ## the mldc data is X,Y,Z tdi
        times, h1, h2, h3 = hoft[0:end_idx, 0], hoft[0:end_idx, 1], hoft[0:end_idx, 2], hoft[0:end_idx, 3]

        delt = times[1] - times[0]


        ## Check if the requested sampel rate is consistant
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt

        return h1, h2, h3, times

    def process_external_data(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to
        read data. Return frequency domain data. Since this was used
        primarily for the MLDC, this assumes that the data is doppler
        tracking and converts to strain data.
        '''

        h1, h2, h3, self.timearray = self.read_data()

        # Calculate other tdi combinations if necessary.
        if self.params['tdi_lev'] == 'aet':
            h1 = (1.0/3.0)*(2*h1 - h2 - h3)
            h2 = (1.0/np.sqrt(3.0))*(h3 - h2)
            h3 = (1.0/3.0)*(h1 + h2 + h3)

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(h1, h2, h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

        # Convert doppler data to strain if readfile datatype is doppler.
        if self.params['datatype'] == 'doppler':

            # This is needed to convert from doppler data to strain data.
            self.r1, self.r2, self.r3 = self.r1/(4*self.f0.reshape(self.f0.size, 1)), self.r2/(4*self.f0.reshape(self.f0.size, 1)), self.r3/(4*self.f0.reshape(self.f0.size, 1))

        elif self.params['datatype'] == 'strain':
            pass


    def tser2fser(self, h1, h2, h3, timearray):

        '''
        Convert time domain data to fourier domain and return ffts. The convention is that the
        the ffts are divided by the sampling frequency and corrected for windowing. A hann window
        is applied by default when moving to the fourier domain. The ffts are also normalized so that
        thier square gives the PSD.

        Parameters
        -----------
        h1, h2, h3 : float
            time series data for the three input channels

        timearray : float
            times corresponding to data in h1, h2, h3

        Returns
        ---------

        r1, r2, r3 : float
            frequency series data for the three input channels

        fdata : float
            Reference frequency series

        tsegstart : float
            Segmented time array giving segment start points

        tsegmid : float
            Segmented time array giving segment midpoints


        '''

        print ("Calculating fourier spectra... ")
        # Number of segmants
        nsegs = int(np.floor(self.params['dur']/self.params['seglen'])) -1

        Nperseg=int(self.params['fs']*self.params['seglen'])

        '''
        # Apply a cascading low pass filter
        b, a = sg.butter(2, 0.4*self.params['fs']/(self.params['fs']/2),\
                btype='lowpass', output='ba')
        #sos = sg.zpk2sos(zz, pp, kk)

        for ii in range(8):
            print('low pass filtering ...')
            h1 = sg.filtfilt(b, a, h1)
            h2 = sg.filtfilt(b, a, h2)
            h3 = sg.filtfilt(b, a, h3)
        '''

        fftfreqs = xp.fft.rfftfreq(Nperseg, 1.0/self.params['fs'])


        # Map of spectrum
        r1 = xp.zeros((fftfreqs.size, nsegs), dtype='complex')
        r2 = xp.zeros((fftfreqs.size, nsegs), dtype='complex')
        r3 = xp.zeros((fftfreqs.size, nsegs), dtype='complex')


        # Hann Window
        hwin = xp.hanning(Nperseg)
        win_fact = xp.mean(hwin**2)


#        zpad = np.zeros(Nperseg)

        ## Initiate time segment arrays
        tsegstart = xp.zeros(nsegs)
        tsegmid = xp.zeros(nsegs)

        # We will use 50% overlapping segments
        for ii in range(0, nsegs):

            idxmin = int(ii*Nperseg)
            idxmax = idxmin + Nperseg
            idxmid = idxmin + int(Nperseg/2)
            if hwin.size != h1[idxmin:idxmax].size:
                import pdb; pdb.set_trace()
            
            if self.gpu:
                r1 = r1.at[:, ii].set(xp.fft.rfft(hwin*h1[idxmin:idxmax], axis=0))
                r2 = r2.at[:, ii].set(xp.fft.rfft(hwin*h2[idxmin:idxmax], axis=0))
                r3 = r3.at[:, ii].set(xp.fft.rfft(hwin*h3[idxmin:idxmax], axis=0))
                tsegstart = tsegstart.at[ii].set(timearray[idxmin])
                tsegmid = tsegmid.at[ii].set(timearray[idxmid])
            else:
                r1[:, ii] =   xp.fft.rfft(hwin*h1[idxmin:idxmax], axis=0)
                r2[:, ii] =   xp.fft.rfft(hwin*h2[idxmin:idxmax], axis=0)
                r3[:, ii] =   xp.fft.rfft(hwin*h3[idxmin:idxmax], axis=0)

                ## There's probably a more pythonic way of doing this, but it'll work for now.
                tsegstart[ii] = timearray[idxmin]
                tsegmid[ii] = timearray[idxmid]

        # "Cut" to desired frequencies
        idx = xp.logical_and(fftfreqs >=  self.params['fmin'] , fftfreqs <=  self.params['fmax'])

        # Output arrays
        fdata = fftfreqs[idx]


        # Get desired frequencies only
        # We want to normalize ffts so thier square give the psd
        # win_fact is to adjust for hann windowing, sqrt(2) for single sided
        r1 = xp.sqrt(2/win_fact)*r1[idx, :]/(self.params['fs']*xp.sqrt(self.params['seglen']))
        r2 = xp.sqrt(2/win_fact)*r2[idx, :]/(self.params['fs']*xp.sqrt(self.params['seglen']))
        r3 = xp.sqrt(2/win_fact)*r3[idx, :]/(self.params['fs']*xp.sqrt(self.params['seglen']))


        np.savez(self.params['out_dir'] + '/' +self.params['input_spectrum'], r1=r1, r2=r2, r3=r3, fdata=fdata)

        return np.array(r1), np.array(r2), np.array(r3), np.array(fdata), np.array(tsegstart), np.array(tsegmid)

    