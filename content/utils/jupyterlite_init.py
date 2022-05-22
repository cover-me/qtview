'''
Initialize JupyterLite for qtview
'''
import os,sys,piplite
from pyodide.http import pyfetch

sys.path.append('./')

async def download_qtview():
    url_base = 'https://raw.githubusercontent.com/cover-me/qtview/main/content/'
    file_list = ['qtview/__init__.py','qtview/data.py','qtview/interact.py',
                 'qtview/operation.py','qtview/plot.py','qtview/jupyterlitetools.py','test.dat']
    for i in file_list:
        # print(i)
        url = url_base + i
        folder, file_name = os.path.split(i)
        if folder and not os.path.exists(folder): os.makedirs(folder)
        response = await pyfetch(url)
        with open(os.path.join(i),'w') as f:
            f.write(await response.string())
            f.close()

await download_qtview()
await piplite.install(['ipywidgets','ipympl'])