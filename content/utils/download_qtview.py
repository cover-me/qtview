'''
Download qtview to JupyterLite's virtual file system
'''
async def download_qtview():
    import os,sys,piplite
    from pyodide.http import pyfetch
    sys.path.append('./')
    
    url_base = 'https://raw.githubusercontent.com/cover-me/qtview/main/content/'
    file_list = ['qtview/__init__.py','qtview/data.py','qtview/interact.py',
                 'qtview/operation.py','qtview/plot.py',
                 'utils/jupyterlitetools.py','utils/download_qtview.py',
                 'test.dat']

    for i in file_list:
        folder, _ = os.path.split(i)
        if folder and not os.path.exists(folder): os.makedirs(folder)

        url = url_base + i
        response = await pyfetch(url)
        with open(os.path.join(i),'w') as f:
            f.write(await response.string())

    await piplite.install(['ipywidgets','ipympl'])
