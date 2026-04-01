# How parameters are calculated

Drfitmap Viewer supports kilosort outputs or spikeinterface outputs directly. Unfortunately,
the term 'amplitude' does not always map to the exact spike times. For kilosort, the term 'amplitude'
are not true amplitudes in uV but XXX, and will change depending on the version.

# Amplitudes

KS1-3
KS4

# Why are whitened templates show for KS output?

For spikeinterface, the template is the . For KS, it is the XXX. It is possible
to unwhiten the templates. However, for KS2.5 and Ks3, this matrix is not save dproperly.
Therefore, for consistent whitetneied templates are always used. If you would like unwhitented
templates, please get in touch.


NOTE THAT SPIKE_CLUSTERS IS ATTEMPTED TO BE USED
NOTE THAT AMPLITUDES ARE ALWAYS POSITIVE
TODO: these amplitudes are not scaled by gain / offset, but this doesn't matter for our purposes
