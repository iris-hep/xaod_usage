---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(ch-configuration)=

# Configuration

In order to run the notebook pages in this book, you'll need to use `pipy` to install a few packages:

```text
jupyter-book
matplotlib
numpy
func-adl-servicex
particle
hist[plot]
```

## Releases

All the plots were produced on the following atlas xAOD release in R21 (Run 2):

```{code-cell}
from func_adl_servicex_xaodr21 import atlas_release
print(atlas_release)
```

Or from the following Run 3 release:

```{code-cell}
from func_adl_servicex_xaodr24 import atlas_release
print(atlas_release)
```
