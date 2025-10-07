import numpy as np
import noise
from PIL import Image, ImageOps
from math import *

def draw_rainbow_line(img, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    length = sqrt(dx * dx + dy * dy)

    # Цвет based on line length and position
    hue = (x0 + y0 + length) % 256
    r = int(128 + 127 * sin(hue * 0.024))
    g = int(128 + 127 * sin(hue * 0.024 + 2))
    b = int(128 + 127 * sin(hue * 0.024 + 4))

    draw_line6(img, x0, y0, x1, y1, color=[r, g, b])

img_mat = np.zeros((600, 800, 3), dtype=np.uint8)

#for i in range(600):
#    for j in range(800):
#        img_mat[i, j] = [0, (i*j)%256, (i*j)%256]
def draw_line(img_mat, x0, y0, x1, y1, color=255):
    count = 10
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def draw_line1(img_mat, x0, y0, x1, y1, color=255):
    count = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def draw_line2(img_mat, x0, y0, x1, y1, color=255):
    for x in range(int(x0), int(x1)):
        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t)* y0 + t*y1)
        img_mat[y, x] = color

def draw_line3(img_mat, x0, y0, x1, y1, color=255):

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1)):
        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t)* y0 + t*y1)
        img_mat[y, x] = color

def draw_line4(img_mat, x0, y0, x1, y1, color=255):

    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1)):

        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)

        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color


def draw_line5(img_mat, x0, y0, x1, y1, color=255):

    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(int(x0), int(x1)):

        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)

        if (xchange):
            img_mat[int(x), int(y)] = color


        else:
            img_mat[int(y), int(x)] = color


        derror += dy
        if (derror > 0.5):
            derror -= 1.0
        y += y_update

def draw_line6(img_mat, x0, y0, x1, y1, color=255):

    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1


    for x in range(x0, x1):

        if (xchange):
            img_mat[x, y] = color

        else:
            img_mat[y, x] = color

        derror += dy
        if (derror > (x1-x0)):
            derror -= 2*(x1-x0)
            y += y_update


def parse_v(f):
    vect = []

    with open(f, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:])
                vect.append((x,y,z))
    return vect

def parse_f(f):
    vect = []

    with open(f, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()
                m = []
                for part in parts[1:]:
                    l = part.split('/')[0]
                    m.append(int(l))
                vect.append(m)
    return vect

for k in range(13):
    x0, y0 = 100, 100
    x1 = 100 - 95*cos(2*pi/13*k)
    y1 = 100 - 95 * sin(2 * pi / 13 * k)
    draw_line6(img_mat, int(x0), int(y0), int(x1), int(y1))

img = Image.fromarray(img_mat, mode='RGB')
#img.save('img6.png')


model = 'model_1.obj'
v = parse_v(model)
f = parse_f(model)
for i, v_ in enumerate(v[:10]):
    print(f"V{i+1}:{v_}")

for i, f_ in enumerate(f[:10]):
    print(f"F{i+1}:{f_}")

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
for i in range(len(v)):
    x = int(5000*v[i][0]+1000)
    y = int(5000*v[i][1]+1000)

    #img_mat[y, x] = [y%255, x%255, 255]

import noise
import numpy as np


def get_perlin_color(x, y, z=0.0, scale=0.01):
    r = noise.pnoise3(x * scale, y * scale, z, octaves=6, persistence=0.5, lacunarity=2.0)
    g = noise.pnoise3(x * scale + 100, y * scale, z, octaves=6, persistence=0.5, lacunarity=2.0)
    b = noise.pnoise3(x * scale, y * scale + 100, z, octaves=6, persistence=0.5, lacunarity=2.0)

    r = int((r + 1) * 127.5)
    g = int((g + 1) * 127.5)
    b = int((b + 1) * 127.5)

    return [r, g, b]


def draw_perlin_rainbow_line(img, x0, y0, x1, y1, time=0.0):
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0

    color = get_perlin_color(mid_x, mid_y, time)
    draw_line6(img, x0, y0, x1, y1, color=color)

for face in f:
    for i in range(len(face)):

        start_idx = face[i] - 1
        end_idx = face[(i + 1) % len(face)] - 1

        x0 = int(9000 * v[start_idx][0] + 1000)
        y0 = int(9000 * v[start_idx][1] + 1000)
        x1 = int(9000 * v[end_idx][0] + 1000)
        y1 = int(9000 * v[end_idx][1] + 1000)

        #draw_line6(img_mat, x0, y0, x1, y1)
        draw_perlin_rainbow_line(img_mat, x0, y0, x1, y1)
        #draw_rainbow_line(img_mat, x0, y0, x1, y1)

img = Image.fromarray(img_mat, mode='RGB')

img = ImageOps.flip(img)
img.save('img_model.png')