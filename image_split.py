import io
import os
from PIL import Image
import numpy as np
CHAR = [str(i) for i in range(10)] + [chr(i)
                                      for i in range(ord('a'), ord('z') + 1)]
cnt = {}
for i in CHAR:
    cnt[i] = 0
target,source = '',''

def depoint(img):  # input: gray image
    pixdata = img.load()
    w, h = img.size
    for i in [0, h - 1]:
        for j in range(w):
            pixdata[j, i] = 255
    for i in [0, w - 1]:
        for j in range(h):
            pixdata[i, j] = 255
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            count = 0
            if pixdata[x, y - 1] > 245:
                count = count + 1
            if pixdata[x, y + 1] > 245:
                count = count + 1
            if pixdata[x - 1, y] > 245:
                count = count + 1
            if pixdata[x + 1, y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x, y] = 255
    return img


def handle_image(path):
    with open(os.path.join(source,path), 'rb') as f:
        pic = f.read()
    pic = io.BytesIO(pic)
    pic = Image.open(pic).convert('1')
    pic = depoint(pic)
    y_min, y_max = 0, 22
    split_lines = [5, 17, 29, 41, 53]
    images = [pic.crop([u, y_min, v, y_max])
              for u, v in zip(split_lines[:-1], split_lines[1:])]
    labels = path[:path.find('.')]
    for i in range(4):
        images[i].save(os.path.join(target,labels[i],'%d.png'%cnt[labels[i]]))
        cnt[labels[i]] += 1


if __name__ == '__main__':
    import sys
    target = sys.argv[2]
    source = sys.argv[1]
    for i in CHAR:
        try:
            os.mkdir(os.path.join(target,i))
        except:
            pass
    for i in os.listdir(os.path.join(source)):
        print(i)
        handle_image(i)
    os.removedirs(os.path.join(target,'o'))
    os.removedirs(os.path.join(target,'z'))
    os.removedirs(os.path.join(target,'9'))