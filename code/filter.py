from PIL import Image, ImageFilter
import numpy as np

img = Image.open("img\masp.jpg")

levels = 7

filtered = [img]
for i in range(levels - 1):
    f = filtered[-1]
    filtered.append(f.filter(ImageFilter.GaussianBlur(5)))

for i, fil in enumerate(filtered):
    fil.save(f"runs/{i}.jpg")

for i in range(len(filtered) - 1):
    f1 = filtered[i].convert('L')
    f2 = filtered[i+1].convert('L')
    diff = np.array(f1, np.int32) - np.array(f2, np.int32)

    # diff = 255 * (diff - np.min(diff)) / np.max(diff)

    Image.fromarray(np.abs(diff).astype(np.uint8)).save(f"runs/g{i}.jpg")






