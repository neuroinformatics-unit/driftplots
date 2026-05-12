# driftplots

`driftplots` is a package for plotting drift maps from Kilosort (1-4) or SpikeInterface's `SortingAnalyzer`.

Interactive mode (above) allows selection of spikes to view the associated template, and
is useful for checking the alignment of two sorted sessions:

<p align="center">
  <img src="https://github.com/user-attachments/assets/b2acde53-1744-4280-8ef2-d9c01b003c92" width="700" />
</p>

Static matplotlib figures are also supported, and come with a range of options
for customising the displayed data and plots:

<p align="center">
  <img width="562" height="370" alt="matplotlib-example" src="https://github.com/user-attachments/assets/5a13caaa-dbdd-481b-863d-c0e643add070" />
</p>

`driftplots` is currently in **beta** release. Please get in touch if you find any issues.

## Get started

See the [documentation](driftplots.neuroinformatics.dev) and locally runnable
[examples](https://github.com/neuroinformatics-unit/driftplots/tree/main/examples) for full details on how to use `driftplots`.

The package can be installed with:
```
pip install driftplots
```

to download the examples and example data, clone this repository and
install locally when in the project root with:

```
pip install -e .
```
