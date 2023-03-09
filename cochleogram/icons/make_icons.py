import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np

n_turns = 2


def make_tile():
    figure, ax = plt.subplots(1, 1, figsize=(1,1 ))
    figure.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0)
    r1 = plt.Rectangle((0, 0), 20, 20, fc='none', ec='k', linewidth=3)
    ax.add_patch(r1)
    r2 = plt.Rectangle((13, 13), 20, 20, fc='none', ec='k', linewidth=3)
    ax.add_patch(r2)
    ax.axis('scaled')
    ax.axis('off')
    figure.savefig('tile.png', transparent=True, bbox_inches='tight')


def make_spiral():
    theta = np.linspace(0, n_turns*2*np.pi, 1000)
    r = theta * 2
    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1 ))
    figure.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0)
    ax.plot(theta, r, 'k-', lw=3, solid_capstyle='round')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rmax(r.max()*1.05)
    ax.axis('off')
    figure.savefig('spiral.png', transparent=True, bbox_inches='tight')


def make_cells():
    theta = np.linspace(0, np.pi/2, 9)
    r = np.ones_like(theta)
    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1 ))
    figure.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0)
    ax.plot(theta, r, 'ko', clip_on=False)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rmax(1.05)
    figure.savefig('cells.png', transparent=True, bbox_inches='tight')


def make_exclude():
    theta = np.linspace(0, np.pi/2, 500)
    r1 = np.ones_like(theta)
    r2 = np.ones_like(theta) * 0.5

    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1))
    figure.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0)
    ax.fill_between(theta, r1, r2, color='k')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.axis('off')
    ax.set_rmax(1.05)
    figure.savefig('exclude.png', transparent=True, bbox_inches='tight')


def make_main_icon():
    spline_effect = [
        pe.Stroke(linewidth=6, foreground="white"),
        pe.Normal(),
    ]
    theta = np.linspace(0, n_turns*2*np.pi, 1000)
    r = theta * 2
    figure, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(1,1 ))
    figure.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0)
    ax.plot(theta, r, 'k-', lw=3, solid_capstyle='round', path_effects=spline_effect)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rmax(r.max()*1.1)
    ax.axis('off')
    figure.savefig('main-icon.png', transparent=True, bbox_inches='tight')


make_spiral()
make_cells()
make_exclude()
make_tile()
make_main_icon()
