"""
Script for create an animation to demonstrate the Secant method for solving a minimimzation problem.

Author: Sivakumar Balasubramanian
Date: 02 March 2024
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rc('font',**{'family':'sans-serif', 'sans-serif': 'Arial'})
mpl.rcParams['toolbar'] = 'None' 

# Supporting functions   
def f(x): return np.polyval([0.05, 0, 0.2, 0, 0], x) - 0.25 * np.sin(2 * x) + 1
def df(x): return np.polyval([0.2, 0, 0.4, 0], x) - 0.5 * np.cos(2 * x)
def update_soln(x0, x1): 
     _f0, _f1 = df(x0), df(x1)
     if _f0 == _f1: return np.nan
     return (_f1 * x0 - _f0 * x1) / (_f1 - _f0)


# Plot update function
def plot_func(ax, axins, x_k):
    x = np.linspace(-6, 6, 1000)
    # Main function.
    ax.plot(x, f(x), lw=4, color="tab:blue", alpha=0.4)
    ax.set_xlim(-6, 12)
    ax.set_ylim(-5, 40)
    ax.tick_params(axis='both', labelsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('axes', -0.03))
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.set_title(r"$f(x) = 0.05 x^4 + 0.2x^2 - 0.25\sin (2x) + 1$" + "\n Newton's Method", fontsize=16)

    # Quadrating approximation at xk
    _x0, _x1 = xk[-2], xk[-3]
    _f0, _df0, _df1 = f(_x0), df(_x0), df(_x1)
    _coeff = [0.5 * (_df1 - _df0) / (_x1 - _x0), _df0, _f0]
    ax.plot(x, np.polyval(_coeff, (x - _x0)), 'tab:red', lw=1)
    ax.plot([_x0, _x0], [f(_x0), 0], color='gray', linestyle='dotted', lw=1)
    ax.plot([_x1, _x1], [f(_x1), 0], color='gray', linestyle='dotted', lw=1)
    ax.plot([xk[-1], xk[-1]], [f(xk[-1]), 0], color='gray', linestyle='dashed', lw=1)

    # Plot points
    ax.plot(xk[-3], f(xk[-3]), 'o', color='tab:purple', markersize=8)
    ax.plot(xk[-2], f(xk[-2]), 'o', color='tab:green', markersize=8)
    ax.plot(xk[-1], f(xk[-1]), 's', color='tab:red', markersize=8)

    # Plot enite iteration history
    ax.plot(xk, np.zeros(len(xk)), 'o', color='black', markersize=5)

    # Display the current solution.
    ax.text(12, 38, f'$x_{{{len(x_k) - 1}}}$ = ' + f'{xk[-1]:+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 35, f'$x_{{{len(x_k) - 2}}}$ = ' + f'{xk[-2]:+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 32, f'$x_{{{len(x_k) - 3}}}$ = ' + f'{xk[-3]:+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 29, f'$f(x_{{{len(x_k) - 2}}})$ = ' + f'{f(xk[-2]):+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 26, f'$f\'(x_{{{len(x_k) - 2}}})$ = ' + f'{df(xk[-2]):+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 23, f'$f\'(x_{{{len(x_k) - 3}}})$ = ' + f'{df(xk[-3]):+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    
    # Update the inset plot.
    axins.plot(np.arange(len(xk)), xk, lw=1, color="tab:green")
    axins.set_xlim(0, 40)
    axins.set_ylim(-10, 10)
    axins.spines["right"].set_visible(False)
    axins.spines["top"].set_visible(False)
    axins.spines["bottom"].set_position("zero")
    axins.spines["left"].set_position(("axes", -0.05))
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')
    axins.tick_params(axis='both', colors='#bbbbbb')


# Event handling
def on_press(event):
    global xk
    ax.cla()
    axins.cla()
    if event.key == 'right':
        # Update solution.
        _xnew = update_soln(xk[-1], xk[-2])
        if np.isnan(_xnew) == False:
            xk.append(_xnew)
        plot_func(ax, axins, xk)
        fig.canvas.draw()
    elif event.key == 'r' or event.key == 'R':
        # Reset search
        xk = [np.random.rand(1)[0] * 12 - 6,]
        xk.append(xk[-1] + 0.1 * np.random.rand(1)[0])
        # Update solution
        _xnew = update_soln(xk[-1], xk[-2])
        if np.isnan(_xnew) == False:
            xk.append(_xnew)
        plot_func(ax, axins, xk)
        fig.canvas.draw()


if __name__ == "__main__":
    # Fixing random state for reproducibility
    xk = [-4.2, -4]

    # Create the figure and the axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    axins = inset_axes(ax, width="30%", height="30%", loc=4, borderpad=1)

    fig.canvas.manager.set_window_title('ALADA Optimization Animations')
    fig.canvas.mpl_connect('key_press_event', on_press)

    # Update solution.
    _xnew = update_soln(xk[-1], xk[-2])
    print(_xnew, np.isnan(_xnew))
    if np.isnan(_xnew) == False:
        xk.append(_xnew)
    plot_func(ax, axins, xk)

    plt.show()
