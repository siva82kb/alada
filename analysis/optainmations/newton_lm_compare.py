"""
Script for create an animation to demonstrate and compare the Newton's method and the Levenberg-Marquardt method for solving a minimimzation problem with a difficult problem where the method fails.

Author: Sivakumar Balasubramanian
Date: 13 March 2024
"""


import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rc('font',**{'family':'sans-serif', 'sans-serif': 'Arial'})
mpl.rcParams['toolbar'] = 'None' 

# Supporting functions   
def f(x): return 1 - np.exp(-np.power(x, 2) / 4)
def df(x): return 0.5 * x * np.exp(-np.power(x, 2) / 4)
def d2f(x): return 0.5 * np.exp(-np.power(x, 2) / 4) - 0.25 * np.power(x, 2) * np.exp(-np.power(x, 2) / 4)


def reset_params():
    global Xk
    # Reset the solution
    _x = np.random.rand(1) * 8  - 4
    Xk = np.array([_x, _x])


def update():
    global Xk
    _xold = Xk[:, -1]
    return np.array([
        [
            _xold[0] - df(_xold[0]) / d2f(_xold[0]),
            _xold[1] - df(_xold[1]) / (d2f(_xold[1]) + l),
        ]
    ]).T


def plot_func():
    # Generate data for plotting
    _xlim = 10
    x = np.linspace(-_xlim, _xlim, 500)
    y = f(x)

    # Plot the search trajectory
    ax.plot(x, y, 'tab:blue', lw=2.0, alpha=0.3)
    ax.plot(Xk[0, 0], f(Xk[0, 0]), 'black', marker='s', markersize=10)
    ax.plot(Xk[1, -1], f(Xk[1, -1]), 'tab:green', marker='o', markersize=10)
    ax.plot(Xk[1, :], f(Xk[1, :]), 'tab:green', lw=1, alpha=1)
    ax.plot(Xk[0, -1], f(Xk[0, -1]), 'tab:red', marker='o', markersize=10)
    ax.plot(Xk[0, :], f(Xk[0, :]), 'tab:red', lw=1, alpha=1)
    ax.plot(0, f(0), 'black', marker='*', markersize=10)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.spines['left'].set_color('#bbbbbb')

    # Set xlims and ylims
    ax.set_xlim(-_xlim, _xlim)
    ax.set_ylim(-0.2, 1.2)

    # Add labels and title
    ax.set_title(r"$f(x) = 1 - \exp\left(-\frac{x^2}{4}\right)$", fontsize=18)


def plot_dist_to_min():
    axins.cla()
    # Update the inset plot.
    _k = Xk.shape[1]
    axins.plot(np.arange(_k), Xk[0, :], lw=1, color="tab:red")
    axins.plot(np.arange(_k), Xk[1, :], lw=1, color="tab:green")
    axins.set_xlim(0, (_k // 40 + 1) * 40)
    axins.set_ylim(-10.1, 10.1)
    axins.set_title(r"$x_k$ vs. $k$", fontsize=14)
    axins.spines["right"].set_visible(False)
    axins.spines["top"].set_visible(False)
    axins.spines["bottom"].set_position("zero")
    axins.spines["left"].set_position(("axes", -0.05))
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')
    axins.tick_params(axis='both', colors='#bbbbbb')


def update_text():
    # Remove the previous text
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)
    k = Xk.shape[1]

    # Method details.
    xpos, ypos, delypos = 0.1, 1.1, 0.1 
    j = 0
    ax2.text(xpos, ypos - j * delypos, f"Netwon's Method",
             fontsize=18, backgroundcolor='tab:red', color='white')
    
    # Function value and its derivatives
    j += 1.2
    _xk = Xk[0, -1]
    ax2.text(xpos, ypos - j * delypos, f"$x_{{{k}}}$ = " + f"{_xk:0.3f}", 
             fontsize=18, color="tab:red")
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$f\\left(x_{{{k}}}\\right)$ = " + f"{f(_xk):0.3f}", 
             fontsize=18, color="tab:red")
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$f'\\left(x_{{{k}}}\\right)$ = " + f"{df(_xk):0.3f}", 
             fontsize=18, color="tab:red")
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$f''\\left(x_{{{k}}}\\right)$ = " + f"{d2f(_xk):0.3f}", 
             fontsize=18, color="tab:red")
    
    j += 1.5
    ax2.text(xpos, ypos - j * delypos, f"Levenberg-Marquardt Method",
             fontsize=18, backgroundcolor='tab:green', color='white')
    # Function value and its derivatives
    j += 1.2
    _xk = Xk[1, -1]
    ax2.text(xpos, ypos - j * delypos, f"$x_{{{k}}}$ = " + f"{_xk:0.3f}", 
             fontsize=18, color="tab:green")
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$f\\left(x_{{{k}}}\\right)$ = " + f"{f(_xk):0.3f}", 
             fontsize=18, color="tab:green")
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$f'\\left(x_{{{k}}}\\right)$ = " + f"{df(_xk):0.3f}", 
             fontsize=18, color="tab:green")
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$f''\\left(x_{{{k}}}\\right)$ = " + f"{d2f(_xk):0.3f}", 
             fontsize=18, color="tab:green")
    
    # Minimum point
    j += 1.3
    ax2.text(xpos, ypos - j * delypos, f"$x^*$ = 0.0", fontsize=18)


# Handling key press events
def on_press(event):
    global funclass, methodclass, Xk, ak, mu

    # Close figure if escaped.
    if event.key == 'escape':
        plt.close(fig)
        return

    # Chekc if the solution needs to be updated.
    if event.key == 'right':
        # Compute the next step        
        # Update the solution
        Xk = np.hstack((Xk, update()))
        # Clear axis
        ax.cla()
    elif event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Save plot
    if event.key == 'ctrl+s':
        fig.savefig("newton_lm_compare.png", dpi=300, bbox_inches='tight')
        fig.savefig("newton_lm_compare.pdf", bbox_inches='tight')
    
    # Draw the plot and text
    update_text()
    plot_func()
    plot_dist_to_min()
    fig.canvas.draw()


if __name__ == "__main__":
    # Create the figure and the axis.
    fig = plt.figure(figsize=(14, 7.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    axins = inset_axes(ax2, width="80%", height="40%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Gradient Descent')

    # Initialize the solution
    l = 1
    Xk = None
    reset_params()
    update_text()
    plot_func()
    plot_dist_to_min()

    # Create the figure and the axis.
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Gradient Descent')
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.tight_layout(pad=3)
    plt.show()
