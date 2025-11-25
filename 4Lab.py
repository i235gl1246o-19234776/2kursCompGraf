import math

import noise
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
    denom = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if abs(denom) < 1e-10:
        return -1.0, -1.0, -1.0

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2

zBuffer = np.full((2000, 2000), np.inf, dtype=np.float32)

def draw_triangle(img_mat, X, Y, Z, color=255):


        x0, x1, x2 = X
        y0, y1, y2 = Y
        z0, z1, z2 = Z

        h, w = img_mat.shape[:2]

        xmin = max(0, min(x0, x1, x2))
        xmax = min(w - 1, max(x0, x1, x2))
        ymin = max(0, min(y0, y1, y2))
        ymax = min(h - 1, max(y0, y1, y2))

        if abs((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) < 1e-10:
            return

        for x in range(xmin, xmax+1):
            for y in range(ymin, ymax+1):
                lambda0, lambda1, lambda2 = baricentric(x, y, x0, y0, x1, y1, x2, y2)
                if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0 and (z0 > 0 and z1 > 0 and z2 > 0):
                    z_ = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                    if z_ < zBuffer[y][x]:
                        zBuffer[y][x] = z_
                        img_mat[y, x] = color
                        #print(f'Нарисовал точку{x,y}')
                        #print(color)

def draw_triangle_gouro(img_mat, X, Y, Z, I0, I1, I2, u0, v0):
    x0, x1, x2 = X
    y0, y1, y2 = Y
    z0, z1, z2 = Z
    if z0 <= 0 or z1 <= 0 or z2 <= 0: #должны быть выше 0
        return


    h, w = img_mat.shape[:2]

    xmin = max(0, min(int(x0), int(x1), int(x2)))
    xmax = min(w-1, max(int(x0), int(x1), int(x2)))
    ymin = max(0, min(int(y0), int(y1), int(y2)))
    ymax = min(h-1, max(int(y0), int(y1), int(y2)))

    lambda0, lambda1, lambda2 = baricentric(xmin, ymin, x0, y0, x1, y1, x2, y2)
    if lambda0 == -1.0:
        return

    lambda0_dx = (y1 - y2) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1_dx = (y2 - y0) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2_dx = lambda0_dx - lambda1_dx

    lambda0_dy = (x2 - x1) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1_dy = (x0 - x2) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2_dy = lambda0_dy - lambda1_dy

    lambda0_new = lambda0 + (xmin - x0) * lambda0_dx + (ymin - y0) * lambda0_dy
    lambda1_new = lambda1 + (xmin - x0) * lambda1_dx + (ymin - y0) * lambda1_dy
    lambda2_new = lambda2 + (xmin - x0) * lambda2_dx + (ymin - y0) * lambda2_dy

    for y in range(ymin, ymax + 1):
        lambda0_r = lambda0_new
        lambda1_r = lambda1_new
        lambda2_r = lambda2_new
        for x in range(xmin, xmax + 1):
            if lambda0_r >= 0 and lambda1_r >= 0 and lambda2_r >= 0:
                z_i = lambda0_r * z0 + lambda1_r * z1 + lambda2_r * z2

                if z_i < zBuffer[y, x]:
                    zBuffer[y, x] = z_i

                    I_f = -255 * (lambda0_r * I0 + lambda1_r * I1 + lambda2_r * I2)
                    if not math.isnan(I_f):
                        color_val = max(0, min(255, int(I_f)))
                        img_mat[y, x] = color_val


            lambda0_r += lambda0_dx
            lambda1_r += lambda1_dx
            lambda2_r += lambda2_dx

        lambda0_new += lambda0_dy
        lambda1_new += lambda1_dy
        lambda2_new += lambda2_dy

def normal(X, Y, Z):
    x0, x1, x2 = X
    y0, y1, y2 = Y
    z0, z1, z2 = Z


    v1 = [x1-x2, y1-y2, z1-z2]
    v2 = [x1-x0, y1-y0, z1-z0]
    n = np.cross(v1, v2)
    return n


def calculate_lighting_gouro(vert_norms, light_dir=np.array([0, 0, 1]), dtype = np.float32):
    vert_norm = vert_norms / np.linalg.norm(vert_norms)
    vertex_lighting = np.dot(vert_norm, light_dir)
    return vertex_lighting

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

def transform_vertex(V, alpha, beta, gamma, tx, ty, tz):

    x, y, z = V

    rz = np.array([
        [np.cos(gamma), np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0,0,1]
    ])

    ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])

    R = rx @ ry @ rz

    v = np.array([x, y, z])

    v_r = R @ v

    v_t = v_r + np.array([tx,ty,tz])

    return v_t[0], v_t[1], v_t[2]

def projector_mutex(X, Y, Z, u0, v0):

    x0, x1, x2 = X
    y0, y1, y2 = Y
    z0, z1, z2 = Z

    ax = 15000
    ay = 15000

    x0_n = (ax * x0) / z0 + u0
    y0_n = (ay * y0) / z0 + v0
    x1_n = (ax * x1) / z1 + u0
    y1_n = (ay * y1) / z1 + v0
    x2_n = (ax * x2) / z2 + u0
    y2_n = (ay * y2) / z2 + v0

    x0_n = int(x0_n)
    y0_n = int(y0_n)
    x1_n = int(x1_n)
    y1_n = int(y1_n)
    x2_n = int(x2_n)
    y2_n = int(y2_n)

    return [x0_n, x1_n, x2_n], [y0_n, y1_n, y2_n], [z0, z1, z2]

model = 'model_1.obj'
v = parse_v(model)
f = parse_f(model)

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)

alpha = np.radians(30)
beta = np.radians(120)
gamma = np.radians(0)

u0 = 1000
v0 = 1000

c_x = sum(vi[0] for vi in v) / len(v)
c_y = sum(vi[1] for vi in v) / len(v)
c_z = sum(vi[2] for vi in v) / len(v)

tx = -c_x
ty = -c_y
tz = -c_z
tz += 2

v_t = []
for x,y,z in v:
    x_n, y_n, z_n = transform_vertex([x,y,z], alpha, beta, gamma, tx, ty, tz)
    v_t.append((x_n, y_n, z_n))
v = v_t
n_all=[]
X_all = []
Y_all = []
Z_all = []
v_n=[[0,0,0] for i in range(len(v))]

for face in f:
    for i in range(len(face)):

        x0 = v[face[i] - 1][0]
        y0 = v[face[i] - 1][1]
        z0 = v[face[i] - 1][2]
        x1 = v[face[(i + 1) % len(face)] - 1][0]
        y1 = v[face[(i + 1) % len(face)] - 1][1]
        z1 = v[face[(i + 1) % len(face)] - 1][2]
        x2 = v[face[2] - 1][0]
        y2 = v[face[2] - 1][1]
        z2 = v[face[2] - 1][2]

        X = [x0, x1, x2]
        Y = [y0, y1, y2]
        Z = [z0, z1, z2]

        n = normal(X, Y, Z)  # нормаль к полигону

        n_all.append(n)
        X_all.append(X)
        Y_all.append(Y)
        Z_all.append(Z)

        v_n[face[0] - 1][0] += n[0]
        v_n[face[0] - 1][1] += n[1]
        v_n[face[0] - 1][2] += n[2]

        v_n[face[1] - 1][0] += n[0]
        v_n[face[1] - 1][1] += n[1]
        v_n[face[1] - 1][2] += n[2]

        v_n[face[2] - 1][0] += n[0]
        v_n[face[2] - 1][1] += n[1]
        v_n[face[2] - 1][2] += n[2]


#нормализуем z
for i in range(len(v_n)):
    len_v_n = (v_n[i][0]**2 + v_n[i][1]**2+v_n[i][2]**2)**0.5
    v_n[i][0] /= len_v_n
    v_n[i][1] /= len_v_n
    v_n[i][2] /= len_v_n
idx=0

for face in f:
    for i in range(len(face)):

        n = n_all[idx]
        X = X_all[idx]
        Y = Y_all[idx]
        Z = Z_all[idx]

        X_, Y_, Z_ = projector_mutex(X, Y, Z, u0, v0)
        idx_p = idx
        if (face[0] - 1 < len(v_n) and face[1] - 1 < len(v_n) and face[2] - 1 < len(v_n)):
            I0 = calculate_lighting_gouro(np.array(v_n[face[0] - 1]),
                                          light_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32))
            I1 = calculate_lighting_gouro(np.array(v_n[face[1] - 1]),
                                          light_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32))
            I2 = calculate_lighting_gouro(np.array(v_n[face[2] - 1]),
                                          light_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32))

            if not (np.isnan(I0) or np.isnan(I1) or np.isnan(I2)):
                draw_triangle_gouro(img_mat, X_, Y_, Z_, float(I0), float(I1), float(I2), u0, v0)
        idx += 1




img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img_model17.png')