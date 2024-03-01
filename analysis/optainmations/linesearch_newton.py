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
def d2f(x): return np.polyval([0.6, 0, 0.4], x) + 1 * np.sin(2 * x) 
def update_soln(x0): return x0 - df(x0) / d2f(x0)


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
    _f, _df, _d2f = f(xk[-2]), df(xk[-2]), d2f(xk[-2])
    ax.plot(x, np.polyval([0.5 * _d2f, _df, _f], (x - xk[-2])), 'tab:red', lw=1)
    ax.plot([xk[-2], xk[-2]], [f(xk[-2]), 0], color='gray', linestyle='dotted', lw=1)
    ax.plot([xk[-1], xk[-1]], [f(xk[-1]), 0], color='gray', linestyle='dashed', lw=1)

    # Plot points
    ax.plot(xk[-2], f(xk[-2]), 'o', color='tab:green', markersize=8)
    ax.plot(xk[-1], f(xk[-1]), 's', color='tab:red', markersize=8)

    # Plot enite iteration history
    ax.plot(xk, np.zeros(len(xk)), 'o', color='black', markersize=5)

    # Display the current solution.
    ax.text(12, 35, f'$x_{{{len(x_k) - 1}}}$ = ' + f'{xk[-1]:+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 32, f'$f(x_{{{len(x_k) - 1}}})$ = ' + f'{f(xk[-1]):+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 29, f'$f\'(x_{{{len(x_k) - 1}}})$ = ' + f'{df(xk[-1]):+1.8f}',
            fontsize=16, verticalalignment='center',
            horizontalalignment='right')
    ax.text(12, 26, f'$f\'\'(x_{{{len(x_k) - 1}}})$ = ' + f'{d2f(xk[-1]):+1.8f}',
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
        xk.append(update_soln(xk[-1]))
        plot_func(ax, axins, xk)
        fig.canvas.draw()
    elif event.key == 'r' or event.key == 'R':
        xk = [np.random.rand(1)[0] * 12 - 6,]
        xk.append(update_soln(xk[-1]))
        plot_func(ax, axins, xk)
        fig.canvas.draw()


if __name__ == "__main__":
    # Fixing random state for reproducibility
    xk = [-4,]

    # Create the figure and the axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    axins = inset_axes(ax, width="30%", height="30%", loc=4, borderpad=1)

    fig.canvas.manager.set_window_title('ALADA Optimization Animations')
    fig.canvas.mpl_connect('key_press_event', on_press)

    # Update solution.
    xk.append(update_soln(xk[-1]))
    plot_func(ax, axins, xk)

    plt.show()
