# qtlab data viewer

Click the badge.

JupyterLite:

[![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://cover-me.github.io/qtview/lab?path=Interactive%20plot%20demo.ipynb)

Binder (May take minutes to load):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cover-me/qtview/main?labpath=content%2FInteractive%20plot%20demo.ipynb)


# miscellaneous:

To find stored files (Chrome): shift + right-click (or right-click if not in Jupyter) -> inspect, in a line showing "Elements Console Recorder ...", find "Application" (may hide behind the ">>" icon), go to "IndexedDB -> JupyterLite Storage -> files". Default files are not stored to this place unless they are modified. You can clear all files there so default files would appear (or one by one in the file browser).

Files in the file broswer (on the left) and files in the kernel (not visible) do not sync in JupyterLite at this moment.

If we drag and drop files to the file browser in JupyterLite, files would get truncated if they are large.
