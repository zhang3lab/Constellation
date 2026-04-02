# keep_mmap_hot.py
import mmap
import os
import time

PATH = "tmp/non_moe_backbone_cache/weights.bin"
STRIDE = 4096          # page size
SLEEP_SEC = 30         # retouch interval

fd = os.open(PATH, os.O_RDONLY)
size = os.path.getsize(PATH)
mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)

print(f"mapped {size/1024/1024/1024:.2f} GiB from {PATH}")

def touch_all():
    s = 0
    for i in range(0, size, STRIDE):
        s ^= mm[i]
    return s

# first warmup
checksum = touch_all()
print(f"initial warmup done, checksum={checksum}")

while True:
    time.sleep(SLEEP_SEC)
    checksum = touch_all()
    print(f"retouched, checksum={checksum}")
