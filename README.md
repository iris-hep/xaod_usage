# xaod_usage

 Book describing how to use the xAOD ATLAS ServiceX Backend.

 Under development - not at all polished!

This web-book website can be found here: [https://iris-hep.org/xaod_usage/](https://iris-hep.org/xaod_usage/)

## Updating The Book

```bash
jupyter-book build --all ./book
ghp-import -n -p -f ./book/_build/html
```

If you want to use this as a test of servicex and the `xAOD` backend, then first clean the build directory to remove all the cache:

```bash
jupyter-book clean --all ./book
```

And then follow the build instructions above.
