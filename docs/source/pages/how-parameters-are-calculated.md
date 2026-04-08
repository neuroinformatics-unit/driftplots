# How features are calculated

Below we detail how the  features displayed in the plots are calculated. This primarily includes the
amplitudes and templates, with calculations differing between the SortingAnalzyer and difference KS versions

A quick note on terminology. Here a 'spike' is a single action potential. During spike sorting, recorded
spikes are assigned to 'units', representing a putative real neuron. We can look at the spikes by looking at
the spike waveform, as recorded. We can visualise this as a single channel, or acrvoss all channels.
The 'spike waveform' is the recorded waveform of an individual spike. The 'template' is a canonical
waveform for a unit. In SpikeInterface, this is the average of all of the individual waveforms. In Kilosort,
this is typically a pre-generated waveform that is fit to individual spikes by scaling it.

<show a picture>

# Amplitudes

In no cases are the mpliatudes provided sacaled in uV. They are the raw int16 values. Tihs is fine for us because
we are just comparing bewteen sessions

In kilosort, the 'amplitudes.npy' file does not include the true waveform amplitudes (through to peak)
but instead a scaling variable that needs to be applied to the template to best match the spike of interest.

When Kilosort performs spike sorting, it first preprocesses the data, including a 'whitening' step. This step
is applied to all channels and removes inter-channel correlations by application of a whitening matrix. The
whitened data has spikes compressed onto fewer channels and the scaling significantly changed.

Templates are fit to individual spikes on the whitened data. In Kilosort1-3, the scaling applied to the canonical
template to fit the individual spikes is saved in the 'amplitudes.npy` file.' Therefore, to reconstruct the approximation
of the individual spikes, we must apply this scaling to the template, and unwhiten the template (it does not matter
if we scale before or after whitening).

Therefore, to compuite amplitudes in Kilosort1-3, we unwhiten the template and apply the amplitude scaling,
as is done by Nick Steinmentz (XX) and in Phy (XX).

However, in Kilosort2.5 and 3, the true inverse whitening matrix is not saved, and instead it is a
scaling matrix. This would make sense if the templates were saved unwhitened, but visual inspection
and comparison with other kilosort versions indicates they are not. Therefore, the amplitudes
from kilosort2.5 and 3 and likely computed on whitented data and are not comparaible with
other versions and cannot be scaled to true amplitudes with gains / offsets.

In Kilosort4, the output `amplitudes.npy` was changed. It is now the XXX, which is more of a 'power'. It is
not the same as the scalings, which are stored in [this variable]() and used to create the drift map
that KS4 outputs, but this variable is not saved (TODO: CHECK). Therefore, we compute the
amplitudes according to [this thread](), although it is not clear whether this is correct because
the templates are certainly white.

In kilosort, the amplitude are always positive whereas in spikeinterface the amplitudes reflect the
direction fo the spike, and are typically negfative. Therefore spikeinterface amplitudes are set to
positive.

ALso, spikeinterface amplitude is computed as XXX. But kilosort we compuate as peak-to-peak.

# Templates (interactive mode)

# Why are whitened templates show for KS output?

For spikeinterface, the template is the . For KS, it is the XXX. It is possible
to unwhiten the templates. However, for KS2.5 and Ks3, this matrix is not save dproperly.
Therefore, for consistent whitetneied templates are always used. If you would like unwhitented
templates, please get in touch.

# Depths
!!! 
NOTE: interpretability across sessions depends on a lot! versions etc