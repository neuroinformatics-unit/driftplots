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
sessions before inter-session matching (for example, with
[UnitMatch](https://github.com/EnnyvanBeest/UnitMatch)). The above example
shows two chronic Neuropixel recording sessions with a gap of two hours
between them. It is clear the recordings are well aligned, as the driftmaps
and templates look similar across the recording depth.

Matplotlib plots can also be generated to get an overview of the data quality
across many sessions, for example by [collating driftmaps into a PDF](pages/examples/creating-pdf).

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

pages/how-to-use-driftplots
pages/how-parameters-are-calculated
pages/examples/index
pages/api_index
```
