from PIL import Image


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


if __name__ == '__main__':
    import sys
    depoint(Image.open(sys.argv[1]).convert('1')).save('temp.png')
