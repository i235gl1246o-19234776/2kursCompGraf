import math

import numpy as np
from PIL import Image, ImageOps

class Quaternions:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
    def __add__(self, other):
        return Quaternions(self.w+ other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Quaternions(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternions(w, x, y, z)

    def norm(self):
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def conjugate(self):
        return Quaternions(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        n = self.norm() ** 2
        return Quaternions(self.w / n, -self.x / n, -self.y / n, -self.z / n)

def euler_to_quaternion(pitch, yaw, roll):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternions(w=w, x=x, y=y, z=z)



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

def draw_triangle_gouro(img_mat, X, Y, Z, I0, I1, I2):
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

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            lambda0, lambda1, lambda2 = baricentric(x, y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z_i = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z_i < zBuffer[y, x]:
                    zBuffer[y, x] = z_i

                    I_f = lambda0 * I0 + lambda1 * I1 + lambda2 * I2

                    color_val = max(0, min(255, int(255 * I_f)))

                    img_mat[y, x] = [color_val, color_val, color_val]

def draw_triangle_texture(img_mat, X, Y, Z, vt0, vt1, vt2, I0, I1, I2, texture_image, texture_size = (1024,1024)):
    x0, x1, x2 = X
    y0, y1, y2 = Y
    z0, z1, z2 = Z
    if z0 <= 0 or z1 <= 0 or z2 <= 0: #должны быть выше 0
        return

    h, w = img_mat.shape[:2]
    w_tex, h_tex = texture_size

    xmin = max(0, min(int(x0), int(x1), int(x2)))
    xmax = min(w-1, max(int(x0), int(x1), int(x2)))
    ymin = max(0, min(int(y0), int(y1), int(y2)))
    ymax = min(h-1, max(int(y0), int(y1), int(y2)))

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            lambda0, lambda1, lambda2 = baricentric(x, y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z_i = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z_i < zBuffer[y, x]:
                    zBuffer[y, x] = z_i

                    I_f = lambda0 * I0 + lambda1 * I1 + lambda2 * I2
                    u = lambda0 * vt0[0] + lambda1 * vt1[0] + lambda2 * vt2[0]
                    v_coord = lambda0 * vt0[1] + lambda1 * vt1[1] + lambda2 * vt2[1]

                    tex_x = int(u * (w_tex - 1))
                    tex_y = int((1 - v_coord)*(h_tex - 1))

                    color_val = I_f*texture_image[tex_y * w_tex + tex_x]
                    img_mat[y, x] = color_val

def normal(X, Y, Z):
    x0, x1, x2 = X
    y0, y1, y2 = Y
    z0, z1, z2 = Z


    v1 = [x1-x2, y1-y2, z1-z2]
    v2 = [x1-x0, y1-y0, z1-z0]
    n = np.cross(v2, v1) #было np.cross(v1,v2), применяем логику алгема
    norm_val = np.linalg.norm(n)
    if norm_val > 0:
        n = n / norm_val
    return n

def calculate_lighting_gouro(vert_norms, light_dir=np.array([0, 0, 1])):
    if np.linalg.norm(vert_norms) > 0:
        vert_norm = vert_norms / np.linalg.norm(vert_norms)

    light_dir = light_dir / np.linalg.norm(light_dir)

    vertex_lighting = np.dot(vert_norm, light_dir)
    return max(0, vertex_lighting)

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

def parse_vt(f):
    vt_list = []
    with open(f, 'r') as file:
        for line in file:
            if line.startswith('vt '):
                parts = line.strip().split()
                u = float(parts[1])
                v = float(parts[2])
                vt_list.append((u, v))
    return vt_list

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

def parse_f_with_texture(f):
    faces_v = []
    faces_vt = []
    with open(f, 'r') as file:
        for line in file:
            if line.startswith('f '):
                parts = line.strip().split()
                v_indices = []
                vt_indices = []
                for part in parts[1:]:
                    data = part.split('/')
                    v_indices.append(int(data[0]))
                    if len(data) > 1 and data[1] != '':
                        vt_indices.append(int(data[1]))
                    else:
                        vt_indices.append(0)
                faces_v.append(v_indices)
                faces_vt.append(vt_indices)
    return faces_v, faces_vt

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

def rotate_point(point, quaternion):
    point_q = Quaternions(0, point[0], point[1], point[2])
    rotated_point = quaternion * point_q * quaternion.conjugate() #Q * v * Q-
    return (rotated_point.x, rotated_point.y, rotated_point.z)

def transform_vertex_Q(v, alpha, beta, gamma, tx, ty, tz):
    q_rotation = euler_to_quaternion(alpha, beta, gamma)
    p = rotate_point(v, q_rotation)
    t_point = (p[0] + tx, p[1] + ty,p[2] + tz)
    return t_point

def projector_mutex(X, Y, Z, u0, v0, ax, ay):

    x0, x1, x2 = X
    y0, y1, y2 = Y
    z0, z1, z2 = Z

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

def delim_na_triangle(face):
    if len(face) == 3:
        return [[face[0] - 1, face[1] - 1, face[2] - 1]]
    faces = []
    for i in range(1, len(face)-1):
        faces.append([face[0]-1, face[i]-1, face[i+1]-1])
    return faces

models = [
    ('12221_Cat_v1_l3.obj', 'Cat_diffuse.jpg'),
    ('12268_banjofrog_v1_L3.obj', '12268_banjofrog_diffuse.jpg')
]

alpha = np.radians(60)
beta = np.radians(150)
gamma = np.radians(90)

ax = 15000
ay = 15000

u0 = 1000
v0 = 1000

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
zBuffer = np.full((2000, 2000), np.inf, dtype=np.float32)

for m, tex in models:
    try:
        texture_img = Image.open(tex).convert('RGB')
        texture_data = np.array(texture_img).reshape(-1, 3)
        texture_size = texture_img.size
        print("Текстура загружена из файла")
    except:
        print("Текстуры нет")

    v = parse_v(m)
    vt_list = parse_vt(m)
        #проверка на vt?
    f,f_vt = parse_f_with_texture(m)

    c_x = sum(vi[0] for vi in v) / len(v)
    c_y = sum(vi[1] for vi in v) / len(v)
    c_z = sum(vi[2] for vi in v) / len(v)


    if m=='12221_Cat_v1_l3.obj':
        tx = -c_x
        ty = -c_y
        tz = -c_z + 700 #2 для зайца, 70 для лягушки, 700 for cat
    else:
        tx = -c_x + 2
        ty = -c_y
        tz = -c_z + 70
    v_t = []
    for x,y,z in v:
        x_n, y_n, z_n = transform_vertex([x,y,z], alpha, beta, gamma, tx, ty, tz)
        v_t.append((x_n, y_n, z_n))
    v = v_t

    v_n=[[0,0,0] for i in range(len(v))]

    counter = 0
    for face in f:
        j = delim_na_triangle(face)
        #print(f"{len(face)}-угольник преобразован в {len(j)} треугольников:", j)
        #counter += 1

        for i in j:
            idx0, idx1, idx2 = [i[0], i[1], i[2]]
            #print(idx0, idx1, idx2)

            x0, y0, z0 = v[idx0]
            x1, y1, z1 = v[idx1]
            x2, y2, z2 = v[idx2]

            X = [x0, x1, x2]
            Y = [y0, y1, y2]
            Z = [z0, z1, z2]

            n = normal(X, Y, Z)

            v_n[idx0] += n
            v_n[idx1] += n
            v_n[idx2] += n
    #нормализуем z
    for i in range(len(v_n)):
        n_val = np.linalg.norm(v_n[i])
        v_n[i][0] /= n_val
        v_n[i][1] /= n_val
        v_n[i][2] /= n_val

    for i, face in enumerate(f):

        j = delim_na_triangle(face)
        for k in j:
            idx0, idx1, idx2 = [k[0], k[1], k[2]]

            x0, y0, z0 = v[idx0]
            x1, y1, z1 = v[idx1]
            x2, y2, z2 = v[idx2]

            X = [x0, x1, x2]
            Y = [y0, y1, y2]
            Z = [z0, z1, z2]

            X_, Y_, Z_ = projector_mutex(X, Y, Z, u0, v0, ax, ay)

            I0 = calculate_lighting_gouro(np.array(v_n[idx0]), np.array([0,0,1]))
            I1 = calculate_lighting_gouro(np.array(v_n[idx1]), np.array([0,0,1]))
            I2 = calculate_lighting_gouro(np.array(v_n[idx2]), np.array([0,0,1]))

            #draw_triangle_gouro(img_mat, X_, Y_, Z_, I0, I1, I2)
            if i < len(f_vt):
                vt_ind = f_vt[i]

                vt0 = vt_list[vt_ind[0] - 1]
                vt1 = vt_list[vt_ind[1] - 1]
                vt2 = vt_list[vt_ind[2] - 1]

                draw_triangle_texture(img_mat, X_, Y_, Z_, vt0, vt1, vt2, I0, I1, I2, texture_data)


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img_model19.png')