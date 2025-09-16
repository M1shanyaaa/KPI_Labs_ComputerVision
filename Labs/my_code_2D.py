import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def rotation_matrix(theta):
    """Матриця обертання (кут у радіанах)"""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def translation_matrix(tx, ty):
    """Матриця перенесення"""
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


my_rectangle = np.array(
    [[0, 0, 4, 4, 0], [0, 3, 3, 0, 0], [1, 1, 1, 1, 1]]  # гомогенна координата
)

figure, axes = plt.subplots()
axes.set_xlim(-30, 30)
axes.set_ylim(-30, 30)
(line,) = axes.plot([], [], lw=2)


def animation_of_slide(i):
    global my_rectangle
    theta = np.radians(i * 3)
    tx = 2
    ty = 2

    transformation = translation_matrix(tx, ty) @ rotation_matrix(theta)
    new_rect = transformation @ my_rectangle
    my_rectangle = new_rect
    line.set_data(new_rect[0, :], new_rect[1, :])
    return (line,)


ani = animation.FuncAnimation(figure, animation_of_slide, frames=100, interval=1000)
plt.show()
