import numpy as np
from PIL import Image, ImageOps
from math import *
import random

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

def baricentric(x,y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2

zBuffer = np.full((2000, 2000), np.inf, dtype=np.float32)

def draw_triangle(img_mat, x0, y0,x1,y1,x2,y2,z0, z1,z2,color=255):
    h, w = img_mat.shape[:2]

    xmin = max(0, min(x0, x1, x2))
    xmax = min(w - 1, max(x0, x1, x2))
    ymin = max(0, min(y0, y1, y2))
    ymax = min(h - 1, max(y0, y1, y2))

    if abs((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))<1e-10:
        return


    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            lambda0, lambda1, lambda2 = baricentric(x,y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z_ = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z_ < zBuffer[y][x]:
                    zBuffer[y][x] = z_
                    img_mat[y, x] = color
                    print(f'Нарисовал точку{x,y}')

def normal(x0, y0,z0,x1,y1,z1,x2,y2,z2):
    v1 = [x1-x2, y1-y2, z1-z2]
    v2 = [x1-x0, y1-y0, z1-z0]
    n = np.cross(v1, v2)
    return n

def cos_sveta(n):
    l=[0,0,1]
    return np.dot(l,n)/(np.linalg.norm(l)*np.linalg.norm(n))



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


model = 'model_1.obj'
v = parse_v(model)
f = parse_f(model)
for i, v_ in enumerate(v[:10]):
    print(f"V{i+1}:{v_}")

for i, f_ in enumerate(f[:10]):
    print(f"F{i+1}:{f_}")

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
'''for i in range(len(v)):
    x = int(5000*v[i][0]+1000)
    y = int(5000*v[i][1]+1000)
    img_mat[y, x] = [y%255, x%255, 255]
    '''

for face in f:
    for i in range(len(face)):

        start_idx = face[i] - 1
        end_idx = face[(i + 1) % len(face)] - 1
        t_idx = face[2] - 1

        x0 = int(9000 * v[start_idx][0] + 1000)
        y0 = int(9000 * v[start_idx][1] + 1000)
        z0 = int(9000 * v[start_idx][2] + 1000)
        x1 = int(9000 * v[end_idx][0] + 1000)
        y1 = int(9000 * v[end_idx][1] + 1000)
        z1 = int(9000 * v[end_idx][2] + 1000)
        x2 = int(9000 * v[t_idx][0] + 1000)
        y2 = int(9000 * v[t_idx][1] + 1000)
        z2 = int(9000 * v[t_idx][2] + 1000)

        #draw_line5(img_mat, x0, y0, x1, y1, [255, 255, 255])
        n=normal(x0,y0,z0, x1,y1,z1, x2,y2,z2)
        p = -255*cos_sveta(n)
        draw_triangle(img_mat, x0, y0, x1, y1, x2, y2, z0, z1, z2,[p, 0, 0])


'''
for k in range(13):
    x0, y0 = 100, 100
    x1 = 100 - 95*cos(2*pi/13*k)
    y1 = 100 - 95 * sin(2 * pi / 13 * k)
    draw_line6(img_mat, int(x0), int(y0), int(x1), int(y1))
'''

#draw_triangle(img_mat, 100,120,  100, 230,300,150, 0, 0, 0)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img_model14.png')