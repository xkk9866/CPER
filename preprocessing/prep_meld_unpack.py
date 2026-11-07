"""Unpack the MELD_raw archive (BigfufuOuO/meld_raw re-pack of the official MELD videos) under $EMO_ROOT/datasets/meld."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import os, tarfile, time
E = _data_root
D = f"{E}/datasets/meld"
t = time.time()
done = f"{D}/MELD_raw.unpacked"
if os.path.exists(done):
    print("already unpacked"); raise SystemExit
with tarfile.open(f"{D}/MELD_raw.tar.gz") as tf:
    n = 0
    for m in tf:
        if ".." in m.name:
            continue
        dst = os.path.join(D, m.name)
        if m.isfile() and os.path.exists(dst) and os.path.getsize(dst) == m.size:
            n += 1
            continue  # resume: already extracted
        for attempt in range(3):  # the NAS occasionally refuses a write (transient)
            try:
                tf.extract(m, D, filter="data"); break
            except OSError as e:
                print("retry", m.name, e, flush=True); time.sleep(5)
        n += 1
        if n % 2000 == 0:
            print(n, m.name, f"{(time.time() - t) / 60:.1f} min", flush=True)
open(done, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
print("MELD unpacked", n, "members", f"{(time.time() - t) / 60:.1f} min", flush=True)
