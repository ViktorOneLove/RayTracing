"""
MIT License
Copyright (c) 2017 Cyrille Rossant
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


class Settings:
    w = 400
    h = 200

    # List of objects.
    color_plane0 = 1. * np.ones(3)
    color_plane1 = 0. * np.ones(3)

    # Light position and color.
    L = np.array([5., 5., -10.])
    color_light = np.ones(3)

    # Default light and material parameters.
    ambient = .05
    diffuse_c = 1.
    specular_c = 1.
    specular_k = 50

    depth_max = 5  # Maximum number of light reflections.

    col = np.zeros(3)  # Current color.
    O = np.array([0., 0.35, -1.])  # Camera.
    Q = np.array([0., 0., 0.])  # Camera pointing to.

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1., -1. / r + .25, 1., 1. / r + .25)

    scene = None


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])


def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(Settings.scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = Settings.scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(Settings.L - M)
    toO = normalize(Settings.O - M)
    # Shadow: find if the point is shadowed or not.
    shadow_coef = 1.
    l = [(intersect(M + N * .0001, toL, obj_sh), k)
         for k, obj_sh in enumerate(Settings.scene) if k != obj_idx]
    if l:
        for sh_int in l:
            if sh_int[0] < np.inf:
                sh_obj = Settings.scene[sh_int[1]]
                shadow_coef *= sh_obj.get('transparency', 0.)
    # Start computing the color.
    col_ray = Settings.ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', Settings.diffuse_c) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', Settings.specular_c) * max(np.dot(N, normalize(toL + toO)),
                                                                0) ** Settings.specular_k * Settings.color_light
    col_ray *= shadow_coef
    return obj, M, N, col_ray


def add_sphere(position, radius, color, reflection=.2,
               transparency=0.0, refraction=0.0):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color),
                reflection=reflection, transparency=transparency,
                refraction=refraction)


def add_plane(position, normal,
              diffuse_c=.75, specular_c=.5, reflection=.25):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (Settings.color_plane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else Settings.color_plane1),
                diffuse_c=diffuse_c, specular_c=specular_c, reflection=reflection)


def refract_ray(vect, n, coeff):
    nv = np.dot(n, vect)
    if nv > 0:
        return refract_ray(vect, n * -1, 1 / coeff)
    a = 1 / coeff
    D = 1 - a * a * (1 - nv * nv)
    if D < 0:
        return None
    b = nv * a + np.math.sqrt(D)
    return (a * vect) - (b * n)


def process_tracing(rayO, rayD, reflection, col, depth, refraction):
    if depth >= Settings.depth_max:
        return

    after_trace = trace_ray(rayO, rayD)
    if not after_trace:
        return

    obj, M, N, col_ray = after_trace
    col += reflection * (1 - obj.get('transparency', 0.)) * col_ray

    # Reflection
    rayO1, rayD1 = M + N * refraction * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
    if process_tracing(rayO1, rayD1, reflection * obj.get('reflection', 1.), col, depth + 1, refraction):
        return

    # Refraction
    ray = refract_ray(rayD, N, obj.get('refraction', 1.))
    if ray is None:
        return
    rayO2, rayD2 = M - N * refraction * .0001, ray
    if process_tracing(rayO2, rayD2, reflection * obj.get('transparency', 1.), col, depth + 1, refraction * (-1)):
        return

    return


def do_refraction(scene):
    Settings.scene = scene

    w = Settings.w
    h = Settings.h

    img = np.zeros((h, w, 3))

    # Loop through all pixels.
    for i, x in enumerate(np.linspace(Settings.S[0], Settings.S[2], w)):
        if i % 10 == 0:
            print(i / float(w) * 100, "%")

        for j, y in enumerate(np.linspace(Settings.S[1], Settings.S[3], h)):
            Settings.col[:] = 0
            Settings.Q[:2] = (x, y)
            D = normalize(Settings.Q - Settings.O)
            depth = 0
            rayO, rayD = Settings.O, D
            reflection = 1.
            refraction = 1
            process_tracing(rayO, rayD, reflection, Settings.col, depth, refraction)
            img[h - j - 1, i, :] = np.clip(Settings.col, 0, 1)

    return img


if __name__ == '__main__':
    pass
