import concurrent.futures, hashlib, json, os, pathlib, threading, time, urllib.request

root = pathlib.Path.home() / '.local/share/lytte-nemotron/models'
path = root / 'NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-IQ4_NL.gguf.part'
state_path = root / 'download-ranges.json'
url = 'https://huggingface.co/sakasegawa/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/resolve/main/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-IQ4_NL.gguf'
size = 18997225856
expected_sha256 = '9688cee08c96ee0f54c0cc1cb07856307a79097fb0088cdb8fb31de6dd26c2be'
lock = threading.Lock()
root.mkdir(parents=True, exist_ok=True)
if path.with_suffix('').exists():
    raise RuntimeError('The completed model already exists; preserving it')
path.touch(exist_ok=True)
if state_path.exists():
    state = json.loads(state_path.read_text())
else:
    start = path.stat().st_size
    step = (size - start + 5) // 6
    state = {'size': size, 'ranges': [{'start': p, 'next': p, 'end': min(p + step, size)} for p in range(start, size, step)]}
    state_path.write_text(json.dumps(state))
if state['size'] != size:
    raise RuntimeError('Unexpected model size in download state')
fd = os.open(path, os.O_WRONLY)

def save():
    tmp = state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state))
    os.replace(tmp, state_path)

def download(item):
    for attempt in range(8):
        with lock:
            start, end = item['next'], item['end']
        if start >= end:
            return
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status != 206 or not content_range.startswith(f'bytes {start}-'):
                    raise RuntimeError(f'Unexpected range response: {response.status} {content_range}')
                offset = start
                saved = start
                while offset < end:
                    block = response.read(min(1024 * 1024, end - offset))
                    if not block:
                        raise IOError('Connection ended early')
                    written = os.pwrite(fd, block, offset)
                    if written != len(block):
                        raise IOError('Short disk write')
                    offset += written
                    if offset - saved >= 16 * 1024 * 1024 or offset == end:
                        with lock:
                            item['next'] = offset
                            save()
                        saved = offset
            print(f'Finished range {item["start"]}-{end}', flush=True)
            return
        except Exception as exc:
            print(f'Retrying range {start}-{end}: {exc}', flush=True)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f'Could not complete range {item["start"]}-{item["end"]}')
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, state['ranges']))
    os.fsync(fd)
finally:
    os.close(fd)
print('Verifying SHA-256...', flush=True)
digest = hashlib.sha256()
with path.open('rb') as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected_sha256:
    raise RuntimeError('Model SHA-256 mismatch; partial file and range state were retained for diagnosis')
path.rename(path.with_suffix(''))
state_path.unlink()
print('Model verified and ready.', flush=True)
