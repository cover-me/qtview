import js, asyncio, os, datetime, base64
from js import Object
from pyodide import to_js
from pyodide.http import pyfetch
DB_NAME = "JupyterLite Storage"

async def content_from_url(url):        
    response = await pyfetch(url)
    if response.status == 200:
        return await response.bytes()

async def list_browser_files():
    
    
    queue = asyncio.Queue(1)
    
    IDBOpenDBRequest = js.self.indexedDB.open(DB_NAME)
    IDBOpenDBRequest.onsuccess = IDBOpenDBRequest.onerror = queue.put_nowait
    
    await queue.get()
    
    if IDBOpenDBRequest.result is None:
        return None
    
    IDBTransaction = IDBOpenDBRequest.result.transaction("files", "readonly")
    IDBObjectStore = IDBTransaction.objectStore("files")
    IDBRequest = IDBObjectStore.getAllKeys()
    IDBRequest.onsuccess = IDBRequest.onerror = queue.put_nowait
    await queue.get()
    
    return IDBRequest.result.to_py() if IDBRequest.result else None

async def content_from_browser(path):
    # https://github.com/innovationOUtside/ouseful_jupyterlite_utils/blob/main/ouseful_jupyterlite_utils/utils.py
    # https://github.com/jupyterlite/jupyterlite/discussions/91#discussioncomment-1137213
    DB_NAME = "JupyterLite Storage"
    
    queue = asyncio.Queue(1)
    
    IDBOpenDBRequest = js.self.indexedDB.open(DB_NAME)
    IDBOpenDBRequest.onsuccess = IDBOpenDBRequest.onerror = queue.put_nowait
    
    await queue.get()
    
    if IDBOpenDBRequest.result is None:
        return None
        
    IDBTransaction = IDBOpenDBRequest.result.transaction("files", "readonly")
    IDBObjectStore = IDBTransaction.objectStore("files")
    IDBRequest = IDBObjectStore.get(path, "key")
    IDBRequest.onsuccess = IDBRequest.onerror = queue.put_nowait
    
    await queue.get()
    
    response = IDBRequest.result.to_py() if IDBRequest.result else None
    if response['format']=='base64':
        file_content = base64.b64decode(response['content'])
    else:
        file_content = response['content'].encode('utf8')
    
    return file_content

async def content_to_browser(content, path, overwrite=False, encode='utf8'):
    # https://github.com/innovationOUtside/ouseful_jupyterlite_utils/blob/main/ouseful_jupyterlite_utils/utils.py
    # via https://github.com/jupyterlite/jupyterlite/discussions/91#discussioncomment-2440964

    path_list = await list_browser_files()# include files in subfolders
    is_file_exist = path in path_list
    
    if is_file_exist and not overwrite:
        print(f'file {path} exists - will not overwrite')
        return
    
    queue = asyncio.Queue(1)
    
    IDBOpenDBRequest = js.self.indexedDB.open(DB_NAME)
    IDBOpenDBRequest.onsuccess = IDBOpenDBRequest.onerror = queue.put_nowait
    await queue.get()
    
    if IDBOpenDBRequest.result is None:
        return None

    IDBTransaction = IDBOpenDBRequest.result.transaction("files", "readwrite")
    IDBObjectStore = IDBTransaction.objectStore("files")

    try:
        # content was a binary stream
        content = content.decode('utf8')
        item_format = 'text'
    except:
        content = base64.b64encode(content).decode('utf8')
        item_format = 'base64'
        
    value = {
        'name': os.path.basename(path), 
        'path': path,
        'format': item_format,
        'created': datetime.datetime.now().isoformat(),
        'last_modified': datetime.datetime.now().isoformat(),
        'content': content,
        'mimetype': 'text/plain',
        'type': 'file',
        'writable': True,
    }

    # see https://github.com/pyodide/pyodide/issues/1529#issuecomment-905819520
    value_as_js_obj = to_js(value, dict_converter=Object.fromEntries)
    
    if is_file_exist:
        IDBRequest = IDBObjectStore.put(value_as_js_obj, path)
    else:
        IDBRequest = IDBObjectStore.add(value_as_js_obj, path)
    IDBRequest.oncomplete = IDBRequest.onsuccess = IDBRequest.onerror = queue.put_nowait
    await queue.get()
    
    return IDBRequest.result

def content_to_mem(content,mem_path,overwrite=False):
    head, tail = os.path.split(mem_path)
    if head == '':
        head = './'
    file_exists = tail in os.listdir(head)
    if file_exists and not overwrite:
        print(f'file {mem_path} exists - will not overwrite')
        return
    with open(mem_path, "wb") as f:
        f.write(content)
        
async def browser_to_mem(b_path,mem_path,overwrite=False):    
    content = await content_from_browser(b_path)
    content_to_mem(content,mem_path,overwrite)
        
async def url_to_mem(url,mem_path,overwrite=False):
    content = await content_from_url(url)
    content_to_mem(content,mem_path,overwrite)

        
async def mem_to_browser(mem_path,b_path,overwrite=False,encode='utf8'):
    with open(mem_path, "rb") as f:
        content = f.read()       
    await content_to_browser(content,b_path,overwrite,encode)

async def url_to_browser(url,b_path,overwrite=False,encode='utf8'):
    content = await content_from_url(url)
    await content_to_browser(content,b_path,overwrite,encode)
    
def ensure_folder_exists(path):
    fold_path = os.path.split(path)[0]
    if fold_path and not os.path.exists(fold_path):
        os.makedirs(fold_path)
