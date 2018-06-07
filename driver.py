import numpy as np
import matplotlib.pyplot as plt
from mydsp.utils import get_corpus
from mydsp.pca import PCA
from mydsp.plots import *

files = get_corpus('woman/ac')
files = sorted(files)[::2]
spectra, taxis, faxis = concatenated_spectrogram(files,10,20) #narrowband
spectrogram(spectra, taxis, faxis)
spectra, taxis, faxis = concatenated_spectrogram(files,5,5) #wideband
spectrogram(spectra, taxis, faxis)

files = get_corpus('woman')
files = sorted(files)
frame_stream = MultiFileAudioFrames(files, 10, 20)

dfts = DFTStream(frame_stream)
dlist = [d for d in dfts]
pca_dfts = PCA(dlist, corr_anal=True)
values = pca_variance_captured(pca_dfts)
values = sorted(values)
#pca_gram(values, taxis)
n1=0
n2=0
n3=0
for i in range(len(values)):
    if np.abs(values[i]) >= 0.1 and not n1:
        n1 = i
    if np.abs(values[i]) >= 0.2 and not n2:
        n2 = i
    if np.abs(values[i]) >= 0.7 and not n3:
        n3 = i

print(n1) 	# capture 10%
print(n2) 	# capture 20%
print(n3)	# capture 70%
