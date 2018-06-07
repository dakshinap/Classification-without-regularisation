'''
Created on Aug 24, 2017

@author: dakshina
'''

import numpy as np
import scipy.signal as signal

class DFTStream:
    '''
    DFTStream - Transform a frame stream to various forms of spectra
    '''


    def __init__(self, frame_stream, specfmt="dB"):
        '''
        DFTStream(frame_stream, specfmt)        
        Create a stream of discrete Fourier transform (DFT) frames using the
        specified sample frame stream. Only bins up to the Nyquist rate are
        returned in the stream.
        
        Optional arguments:
        
        specfmt - DFT output:  
            "complex" - return complex DFT results
             "dB" [default] - return power spectrum 20log10(magnitude)
             "mag^2" - magnitude squared spectrum
        '''

        self.frame_stream = frame_stream
        self.specfmt = specfmt

        self.frame_s = frame_stream.get_framelen_ms()/1000 # frame length
        self.Fs = frame_stream.get_Fs() # sample rate
        self.frame_N =  int(np.round(self.Fs * self.frame_s))# Num samples

        # Use self.bins_N to represent the number of bins returned
        self.bins_Hz = np.arange(self.frame_N) / self.frame_N*self.Fs
        self.bins_N = [i for i in self.bins_Hz]		# no of bins

        self.window = signal.get_window("hamming", self.frame_N)   
        self.frame_it = iter(frame_stream)
        
    def shape(self):
        "shape() - Return dimensions of tensor yielded by next()"
        return np.asarray([len(self.bins_N), 1])
    
    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return np.asarray(np.product(self.shape()))
    
    def get_Hz(self):
        "get_Hz(Nyquist) - Return frequency bin labels"
        return self.bins_Hz
            
    def __iter__(self):
        "__iter__() Return iterator for stream"
        return self
    
    def __next__(self):
        "__next__() Return next DFT frame"
        framedata = next(self.frame_it)
        windowed_x = framedata * self.window
        X = np.fft.fft(windowed_x)
        if self.specfmt is "complex":
            return X
        elif self.specfmt is "dB":
            magX = np.abs(X)
            return 20 * np.log10(magX)
        elif self.specfmt is "mag^2":
            sqMagX = np.square(np.abs(X))
            return 20 * np.log10(sqMagX)
        
    def __len__(self):
        "len() - Number of tensors in stream"
        self.framer = []
        z = []
        for i,f in enumerate(self.frame_stream):
            if i % self.frame_stream.size():
                self.framer.append(z)
                z=[]
            z.append(f)
        self.framer.append(z)
        return len(self.framer)

        
        
        
    
        