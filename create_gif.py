from ray_tracing import *
import imageio
import math


if __name__ == '__main__':
    images_movements = []
    x = -.75
    y = .1
    z = 2.25

    for angle in range(0, 20, 1):
        new_x = x * math.cos(angle) - z * math.sin(angle)
        new_z = x * math.sin(angle) + z * math.cos(angle)

        scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.], transparency=0.01, refraction=0.8),  # blue
                 add_sphere([new_x, .1, new_z], .4, [.5, .223, .5], transparency=0.3, refraction=1),  # pink
                 add_sphere([-2.75, .1, 2.5], .4, [1., .572, .184], transparency=0.2, refraction=0.5),  # orange
                 add_plane([0., -.5, 0.], [0., 1., 0.]),
                 ]

        img = do_refraction(scene)
        images_movements.append(img)
    imageio.mimsave('movie_pink.gif', images_movements, 'GIF', **{'duration': 2})


    num_of_steps = 10
    transparency_start = 0
    transparency_end = 1
    refraction_start = 1
    refraction_end = 2

    transparency_step = abs(transparency_end - transparency_start) / num_of_steps
    refraction_step = abs(refraction_end - refraction_start) / num_of_steps

    transparencies = np.arange(transparency_start, transparency_end, transparency_step)
    refractions = np.arange(refraction_start, refraction_end, refraction_step)

    images_parameters = []
    for transparency, refraction in zip(transparencies, refractions):
        scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.], transparency=0.5, refraction=refraction),  # blue
                 add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5], transparency=transparency, refraction=1.),  # pink
                 add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184], transparency=0.1, refraction=0.5),  # orange
                 add_plane([0., -.5, 0.], [0., 1., 0.]),
                 ]
        img = do_refraction(scene)
        images_parameters.append(img)
    imageio.mimsave('params_pink.gif', images_parameters, 'GIF', **{'duration': 1})