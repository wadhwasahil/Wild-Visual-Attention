from PIL import Image
import os
import math

path = "MSRA-TD500/train/"
def read(path=path):
    gt_meta = []
    img_meta = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".gt"):
                gt_meta.append(file)
            if file.endswith(".JPG"):
                img_meta.append(file)

    gt_meta.sort()
    img_meta.sort()
    for i in range(len(img_meta)):
        try:
            img_path = path + img_meta[i]
            print(img_path)
            gt_path = path + gt_meta[i]
            with open(gt_path) as f:
                lines = f.readlines()
            for line in lines:
                gt = [float(x) for x in line.split()]
                index, difficult_label, x, y, w, h, theta = gt
                print(gt)
                im = Image.open(img_path)
                im = im.rotate(4)
                box = (x, y, x + w, y + h)
                im = im.crop(box)
                print(im.format, im.size, im.mode)
                im.show()
                break
        except:
            print("Can't open {} image".format(img_path))
        if i > 5:
            break

read()