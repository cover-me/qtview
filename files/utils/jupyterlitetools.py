'''
Tools for file and internet operations in JupyterLite
'''
import sys

if 'pyolite' in sys.modules:# if jupyterlite
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

        if IDBRequest.result:
            response = IDBRequest.result.to_py()
        else:
            return None

        if response['format']=='base64':
            file_content = base64.b64decode(response['content'])
        else:
            file_content = response['content'].encode('utf8')

        return file_content

    async def content_to_browser(content, path, overwrite=False, encode='utf8'):
        # https://github.com/innovationOUtside/ouseful_jupyterlite_utils/blob/main/ouseful_jupyterlite_utils/utils.py
        # via https://github.com/jupyterlite/jupyterlite/discussions/91#discussioncomment-2440964

        if content:
            path_list = await list_browser_files()# include files in subfolders
            is_file_exist = path in path_list

            if is_file_exist and not overwrite:
                print(f'File "{path}" already exists, will not overwrite')
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

            if path.startswith('./'):
                path = path[2:]

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

    #         return IDBRequest.result
        else:
            print("Empty content. No file is created.")

    def content_to_mem(content,mem_path,overwrite=False):
        if content:
            head, tail = os.path.split(mem_path)
            if head == '':
                head = './'
            file_exists = tail in os.listdir(head)
            if file_exists and not overwrite:
                print(f'File "{mem_path}" already exists, will not overwrite')
                return
            with open(mem_path, "wb") as f:
                f.write(content)
        else:
            print("Empty content. No file is created.")

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


    async def zenodo_downloader(record_id, file_name, overwrite=True):
        url = f'https://zenodo.org/api/records/{record_id}'
        resp = await pyfetch(url)
        record_meta = await resp.json()
        found = False
        for i in record_meta['files']:
            if i['key'] == file_name:
                file_dict = i
                found = True
                break
        if not found:
            raise Exception('File not found.')
        url = file_dict['links']['self']
        print(f'Size: {file_dict["size"]/1048576} MB, downloading...')
        await download_file(url,file_name,overwrite)
        print('Done!')
        
    async def download_file(url,path,overwrite=False):
        if os.path.exists(path) and not overwrite:
            print(f'File "{mem_path}" already exists, will not overwrite')
            return
        resp = await js.fetch(url)
        file_size = float(resp.headers.get('content-length'))
        reader = resp.body.getReader()
        with open (path,'wb') as f:
            counter = 0
            counter2 = 0
            while 1:
                chunk = (await reader.read()).to_py()
                if chunk['done']:
                    break
                f.write(chunk['value'])
                counter += len(chunk['value'])
                counter2 += 1
                if counter2%100==0:
                    print(f'\r{counter/file_size:.0%}',end='')
            print()
else:
    import requests,os
    def zenodo_downloader(record_id, file_name, overwrite=True):
        url = f'https://zenodo.org/api/records/{record_id}'
        resp = requests.get(url)
        record_meta = resp.json()
        for i in record_meta['files']:
            if i['key'] == file_name:
                file_dict = i
                break
        url = file_dict['links']['self']
        print(f'Size: {file_dict["size"]/1048576} MB, downloading...')
        download_file(url,file_name,overwrite)
        print('Done!')
        
    def download_file(url,path,overwrite=False):
        if os.path.exists(path) and not overwrite:
            print(f'File "{mem_path}" already exists, will not overwrite')
            return
        resp = requests.head(url)
        file_size = float(resp.headers['Content-Length'])/1024
        resp = requests.get(url, stream=True)
        with open (path,'wb') as f:
            counter = 0
            for chunk in resp.iter_content(chunk_size=1024):
                f.write(chunk)
                counter += 1
                if counter%1024 == 0:
                    print(f'\r{counter/file_size:.0%}',end='')
            print()
