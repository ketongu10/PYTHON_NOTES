import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation







def init():
      sns.heatmap(np.zeros((10, 10)), vmax=.8, square=True, cbar=False)

def animate(i, F, dt):
    data = np.sum(F[i*dt], axis=(2,3)).swapaxes(0, 1)
    plot = sns.heatmap(data, square=True, cbar=False)
    plot.invert_yaxis()



def render_animation(F, dt=1):
    fig, ax = plt.subplots()

    anim = animation.FuncAnimation(fig, animate,fargs=(F,dt), frames=len(F)//dt, repeat=False)
    anim.save("../plots/2.gif", writer="ffmpeg", fps=20)

