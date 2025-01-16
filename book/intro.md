# Using the ATLAS xAOD Typed Backend

```{warning}
This documentation book is currently in the proccess of being updated to r22 and ServiceX 3. By using a combination of the updated examples and the old examples an understanding of how to complete the examples using the latest version can be made. For up to date examples look to "The Jet Collection" page.
```

Come here to learn:

- How to setup your environment to access xAOD data in ATLAS (R21)
- Collections from the xAOD that are available
- Access methods, attributes, and decorations on the xAOD Event Data Model (EDM).

This tutorial book takes for granted that you:

- Have access to an ATLAS xAOD ServiceX backend
- Know the basics about how to use ServiceX to filter, select data, etc.
- Basic knowledge of the ATLAS xAOD data model (at the level of a standard ATLAS tutorial)
- The names of the xAOD collections you are interested in, etc.

## Typed?

Normally, one does not think of Python as a `typed` language, like C++ or many other languages.
However, most Python code does not take advantage of this - a variable is always an `int` or a `string`
or similar. _Type Hints_ are the Python way of expressing this. They are a fundamental part of the modern
Python language (see [PEP 482](https://www.python.org/dev/peps/pep-0482/), [PEP 483](https://www.python.org/dev/peps/pep-0483/),
[PEP 484](https://www.python.org/dev/peps/pep-0484/), etc.). Typing offers several advantages:

- Various tools like [`pylance`](https://github.com/microsoft/pylance-release) and [`mypy`](http://mypy-lang.org/) can spot errors before the code is run
- Editors, like [`vscode`](https://code.visualstudio.com/), can give you suggestions motivated by the type definitions

Further, `func_adl` libraries can use this type information to configure the backend C++ on the fly:

- New collections can be made accessible.
- Objects that are returned by various methods can be properly interpreted by the `func_adl` C++ backend.
- Arbitrary C++ code can be downloaded and executed as part of the query to ServiceX.

For example, here is a Jupyter notebook opened in visual studio code, showing the possible collections in an event:

![Example of using VSCode's built in type-checker to predict method names](assets/vscode-intellisense.png)

And here the type checker `pylance` has flagged a bad method name, and a mouse-over shows more details on the error

![Example of using VSCode's pylance type checker to flag an error](assets/vscode-intellisense-error.png)

## Setup and config

Everything in this book can be executed and, at the time when the book was published, ran. For
this to work we need an up-to-date version of ServiceX running. We call this backend `xaod_r21` for
the C++ ATLAS xAOD backend based on the Release 21 of the ATLAS analysis software. As per normal for ServiceX, `xaod_r21` should
be defined in your `servicex.yaml` or `.servicex` file in your `$HOME` directory.

See the chapter on [configuration](ch-configuration) for more information.

## Improvements

The source code for this book can be found in the [xaod_usage](https://github.com/gordonwatts/xaod_usage) github repository. Please feel free to submit a pull request or create an issue for new things that are needed!