import numpy as np
import matplotlib.pyplot as plt

n_turns = 2


def make_spiral():
    theta = np.linspace(0, n_turns*2*np.pi, 1000)
    r = theta * 2
    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1 ))
    ax.plot(theta, r, 'k-', lw=3, solid_capstyle='round')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    figure.savefig('spiral.png', transparent=True)


def make_cells():
    theta = np.linspace(0, np.pi, 8)
    r = np.ones_like(theta)
    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1 ))
    ax.plot(theta, r, 'ko')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rmax(1.25)
    figure.savefig('cells.png', transparent=True)


def make_exclude():
    theta = np.linspace(0, np.pi, 500)
    r1 = np.ones_like(theta)
    r2 = np.ones_like(theta) * 0.5

    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1 ))
    ax.fill_between(theta, r1, r2, color='k')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis('off')
    ax.set_rmax(1.25)
    figure.savefig('exclude.png', transparent=True)

make_spiral()
make_cells()
make_exclude()
