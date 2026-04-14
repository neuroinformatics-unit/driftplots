:html_theme.sidebar_secondary.remove:

# driftplots

`driftplots` is a tool for visualising and saving drift maps from 
Kilosort or SpikeInterface. It can be used to assess data quality through 
static matplotlib figures or an interactive viewer:

```{image} /_static/driftmap-viewer.gif
:align: center
:width: 1000px
```

<br>

Interactive mode can be used to check alignment between recording 
sessions before inter-session matching e.g. with UnitMatch. In the above example,
it is clear the recordings (chronically recorded Neuropixel sessions with a gap 
between them of 2 hours) are well aligned, as the driftmap and templates 
look similar across the recording depth.

Matplotlib mode can also be used to get an overview of the data quality
across many sessions, for example by [collating these into a PDF](pages/examples/creating-pdf).

::::{grid} 1 1 3 3
:gutter: 4

:::{grid-item-card} {fas}`book;sd-text-primary` Using `driftplots`
:link: pages/how-to-use-driftplots
:link-type: doc

Get started with `driftplots`
:::

:::{grid-item-card} {fas}`book;sd-text-primary` Implementation Details
:link: pages/how-parameters-are-calculated
:link-type: doc

Details on how the parameters are calculated.
:::

:::{grid-item-card} {fas}`lightbulb;sd-text-primary` Examples
:link: pages/examples/index
:link-type: doc

Using `driftplots` in practice
:::

:::{grid-item-card} {fas}`code;sd-text-primary` API
:link: pages/api_index
:link-type: doc

Full Python API reference.
:::

::::


```{toctree}
:maxdepth: 2
:caption: index
:hidden:

pages/articles/index
pages/examples/index
pages/api_index
```
