# qtlab data viewer

This is the Jupyter notebook version of qtplot. Click the badges below to see notebooks.

JupyterLite (code run locally in the sandbox of the browser. open immediately, but may take half a minute to download modules when running the first cell):

[![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://cover-me.github.io/qtview/lab?path=Example+interactive+plot.ipynb)

Binder (code run on an online server, may take minutes to initialize, fast after initializated):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cover-me/qtview/main?labpath=content%2FExample%20interactive%20plot.ipynb)


![image](https://user-images.githubusercontent.com/22870592/170893035-82dd74b5-9790-4ab7-b71a-dfcf98386990.png)



# miscellaneous:

To find stored files (Chrome): shift + right-click (or right-click if not in Jupyter) -> inspect, in a line showing "Elements Console Recorder ...", find "Application" (may hide behind the ">>" icon), go to "IndexedDB -> JupyterLite Storage -> files". Default files are not stored to this place unless they are modified. You can clear all files there so default files would appear (or one by one in the file browser).

Files in the file broswer (on the left) and files in the kernel (not visible) do not sync in JupyterLite at this moment.

If we drag and drop files to the file browser in JupyterLite, files would get truncated if they are large.

The scipy module increases the loading time in JupyterLite from 20 s to 30 s.
