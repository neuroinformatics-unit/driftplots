:html_theme.sidebar_secondary.remove:

```{raw} html
<div style="height: 0; visibility: hidden;">
```
# driftmap_viewer
```{raw} html
</div>
```

Driftmap Viewer (TODO: NAME) is a tool to visualise and save drift maps from kilosort
or SpikeInterface and interactively compare these across sessions.

The main use case of Driftmap Viewer is to check the alignment of sessions prior to
performing inter-session Unit Matching or data concatenation. In the below example,
(XXX) it is clear the two sessions are well aligned, with drift maps looking similar and
templates clearly matching at the same position of the probe.

Driftmap viewer can be used in this interactive view () or through matplotlib
plots that can be used to save PDF over an entire experiment for quick checking

::::{grid} 1 1 3 3
:gutter: 4

:::{grid-item-card} {fas}`book;sd-text-primary` Articles
:link: pages/articles/index
:link-type: doc

Guides for the interactive and matplotlib viewers, and how drift-map parameters are calculated.
:::

:::{grid-item-card} {fas}`lightbulb;sd-text-primary` Examples
:link: pages/examples/index
:link-type: doc

Worked examples using ``driftmap_viewer`` in practice.
:::

:::{grid-item-card} {fas}`code;sd-text-primary` API
:link: pages/api_index
:link-type: doc

Full Python API reference.
:::

::::

``driftmap_viewer`` loads Kilosort sorter output and creates
interactive (pyqtgraph) or static (matplotlib) drift map plots,
making it easy to inspect electrode drift across a recording session.

```{toctree}
:maxdepth: 2
:caption: index
:hidden:

pages/articles/index
pages/examples/index
pages/api_index
```
