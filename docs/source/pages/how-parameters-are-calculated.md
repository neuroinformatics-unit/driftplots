# How features are calculated

The key features computed in `driftplots` are the spike amplitudes, depths alongside the displayed
templates. Below are detailed on how these are computed, as the methods vary between
Kilosort, SpikeInterface and different Kilosort versions.


:::{note}
A quick note on terminology:

- *spike* : 'a single action potential as recorded on the probe and detected by the sorter.
- *unit* : a group of spikes, determined by the sorter to come from the same neuron.
- *spike waveform* : the waveform (i.e. voltage signal as detected on the probe) of a single spike.
- *template* : the canonical waveform that represents a unit. Can be conceptualised as the average of
representative spike of a neuron. Individual spike waveforms are similar to their template waveform.
- *whitened template*

:::

TODO: ADD IMAGE

All data is presented as the values on which the data is stored on disk. For NeuroPixels with Kilosort
this is `int16`. The data is not scaled to `uV`.

The templates, amplitudes and spike depths calculated in SpikeInterface and Kilosort
are different.


# Amplitudes

**SpikeInterface**

In SpikeInterface, the spike amplitude is calculated

**Kilosort**

In Kilosort, the `amplitudes.npy` file is not the true spike amplitude. See below for how this
value is used to approximate true spike amplitudes across Kilosort versions.

In Kilosort1-3 it is a scalar
to be applied to the unwhitened template to scale it to best match the spike waveform. Therefore,
the amplitude of the unwhitened template scaled with the `amplitude.npy` value for the spike
of interest should be a good approximation of the true spike amplitude. However, issues
with the saving of the inverse whitening matrix in Kilosort 2.5 and 3 add complications. In Kilosort 4,
`ampltudesn`

*Kilosort 1, 2*

*Kilosort 2.5, 3*

*Kilosort 4*



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
