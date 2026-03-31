# Using the interactive viewer

Driftmap viewer is a lightweight tool. We can create the viewer instance
by supplying a path to a sorting output, or a SpikeInterface sortinganalyzer object.

For example:

```python

```

it just shows the template for that spike, it does NOT show the spike or even the scaled template, 
due to inconsistencies in XXX.

This will load all key features (e.g. spike imes, ampliudes) into memory, once.

Next, we can plot using driftmap interactive or matplotlib. These take all of the same arguments,
except for matpltlib which also can show a 1D activity history.

As part of XXX, the data is processed.

WARNINGS: designed for NP1 probes, will work for Cam Neurotech or NeuroNexus but not well tested. Please get in touch.

# Using the interactive Viewer

Note for SI:
max_spikes_per_unit=1_000_000,  # This determines the number of spikes that will appear on the SI drift plot

# for KS,
# TODO: these amplitudes are not scaled by gain / offset, but this doesn't matter for our purposes

See how amplitudes are calculated
Why uniwhitened templates are displayed (except spikeinterface).

# Using matplotlib


# See the API documentation for each thing
