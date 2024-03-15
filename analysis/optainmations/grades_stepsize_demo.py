"""
Script for create an animation to demonstrate the gradient descent method for 
a univariate minimization problem.

Author: Sivakumar Balasubramanian
Date: 12 March 2024
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import aladaopt

mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
mpl.rcParams['toolbar'] = 'None' 


# Supporting functions   
def f(x): return np.polyval([1, 0, 0], x)
def df(x): return np.polyval([0, 2, 0], x)


def reset_params():
    global Xk
    # Reset the solution
    _x = np.random.rand(1) * 10  - 5
    Xk = np.array([_x, _x, _x, _x])


def update():
    global ak1, ak2, ak3
    _xold = Xk[:, -1]
    return np.array([
        [_xold[0] - ak1 * df(_xold[0]),
         _xold[1] - ak2 * df(_xold[1]),
         _xold[2] - ak3 * df(_xold[2]),
         _xold[3] - ak4 * df(_xold[3])]
    ]).T


def plot_func():
    # Generate data for plotting
    x = np.linspace(-5, 5, 500)
    y = f(x)

    # Plot the search trajectory
    ax.plot(x, y, 'tab:blue', lw=2.0, alpha=0.3)
    ax.plot(Xk[0, 0], f(Xk[0, 0]), 'black', marker='s', markersize=10)
    ax.plot(Xk[3, -1], f(Xk[3, -1]), 'tab:gray', marker='o', markersize=10)
    ax.plot(Xk[3, :], f(Xk[3, :]), 'tab:gray', lw=1, alpha=0.6)
    ax.plot(Xk[2, -1], f(Xk[2, -1]), 'tab:purple', marker='o', markersize=10)
    ax.plot(Xk[2, :], f(Xk[2, :]), 'tab:purple', lw=1, alpha=0.6)
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
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 26)

    # Add labels and title
    ax.set_title(r"$f(x) = x^2$", fontsize=18)


def plot_dist_to_min():
    axins.cla()
    # Update the inset plot.
    _k = Xk.shape[1]
    axins.plot(np.arange(_k), Xk[0, :], lw=1, color="tab:red")
    axins.plot(np.arange(_k), Xk[1, :], lw=1, color="tab:green")
    axins.plot(np.arange(_k), Xk[2, :], lw=1, color="tab:purple", alpha=0.6)
    axins.plot(np.arange(_k), Xk[3, :], lw=1, color="tab:gray", alpha=0.6)
    axins.set_xlim(0, (_k // 40 + 1) * 40)
    axins.set_ylim(-5.1, 5.1)
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
    ax2.text(xpos, ypos - j * delypos, f"Effect of Step Size",
             fontsize=18, backgroundcolor='tab:red', color='white')

    # Iteration
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"Iteration $k = $" + f"{k}",
            fontsize=14)
    
    # Step size
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$\\alpha_1 = $" + f"{ak1:0.3f}",
            fontsize=18, color="tab:red")
    j += 0.9
    ax2.text(xpos, ypos - j * delypos, f"$\\alpha_2 = $" + f"{ak2:0.3f}",
            fontsize=18, color="tab:green")
    j += 0.9
    ax2.text(xpos, ypos - j * delypos, f"$\\alpha_3 = $" + f"{ak3:0.3f}",
            fontsize=18, color="tab:purple")
    j += 0.9
    ax2.text(xpos, ypos - j * delypos, f"$\\alpha_4 = $" + f"{ak4:0.4f}",
            fontsize=18, color="tab:gray")

    
    # Minimum point
    j += 1.3
    ax2.text(xpos, ypos - j * delypos, f"$x^*$ = 0.0", fontsize=18)

    # Current points
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"$x_{{{k}}}$ = " + f"{Xk[0, -1]:0.3f}", fontsize=18, 
             color="tab:red")
    j += 0.9
    ax2.text(xpos, ypos - j * delypos, f"$x_{{{k}}}$ = " + f"{Xk[1, -1]:0.3f}", fontsize=18, 
             color="tab:green")
    j += 0.9
    ax2.text(xpos, ypos - j * delypos, f"$x_{{{k}}}$ = " + f"{Xk[2, -1]:0.3f}", fontsize=18, 
             color="tab:gray")
    j += 0.9
    ax2.text(xpos, ypos - j * delypos, f"$x_{{{k}}}$ = " + f"{Xk[3, -1]:0.3f}", fontsize=18, 
             color="tab:purple")
    


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
        fig.savefig("gradient_descent.png", dpi=300, bbox_inches='tight')
        fig.savefig("gradient_descent.pdf", bbox_inches='tight')
    
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
    axins = inset_axes(ax2, width="80%", height="30%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Gradient Descent')

    # Initialize the solution
    ak1, ak2, ak3, ak4 = 0.005, 0.2, 0.9, 1.02
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