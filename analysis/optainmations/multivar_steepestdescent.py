"""
Script for create an animation to demonstrate steepest descent.

Author: Sivakumar Balasubramanian
Date: 14 March 2024
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import aladaopt

mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
mpl.rcParams['toolbar'] = 'None' 


def reset_params():
    global Xk, tau, ak, k
    # Reset the solution
    Xk = np.random.rand(2, 1) * 8 - 4
    ak = 5
    tau = 0.99
    k = 0


def update():
    global Xk, Xbt, k, ak
    _xk = Xk[:, -1].reshape(-1, 1)
    _grad = funclass.grad(_xk[0, 0], _xk[1, 0])
    ak = exact_search(ak=1, dk=_grad)
    return aladaopt.GradientDescent.update(_xk, ak, _grad)


def get_function_along_dir(xk, dirvec):
    _t = np.linspace(-10, 10, 501)
    _x = xk[0] + _t * dirvec[0]
    _y = xk[1] + _t * dirvec[1]
    return _t, funclass.func(_x, _y)


def get_search_lines(xk, dirvec):
    _x = xk[0] + 10 * np.array([-dirvec[0], dirvec[0]])
    _y = xk[1] + 10 * np.array([-dirvec[1], dirvec[1]])
    return _x, _y


def exact_search(ak, dk):
    global Xk, tau, k
    # Check there is previous backtracked solution.
    _xk = Xk[:, -1].reshape(-1, 1)
    dk = funclass.grad(_xk[0, 0], _xk[1, 0])
    _fk = funclass.func(_xk[0, 0], _xk[1, 0])
    while True:
        # Update the solution
        # k += 1
        _xb = _xk - ak * dk
        # Check current function value is less than the previous one.
        _fb = funclass.func(_xb[0, 0], _xb[1, 0])
        _gb = funclass.grad(_xb[0, 0], _xb[1, 0])
        _gbd = dk.T @ _gb
        if (_fb <= _fk and np.abs(_gbd) <= 1e-1): return ak
        ak *= tau


def update_text():
    global k, ak
    # Remove the previous text
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)

    # Method details.
    xpos, ypos, delypos = 0.1, 1.1, 0.1 
    j = 0
    ax2.text(xpos, ypos - j * delypos, f"Gradient Descent: Backtracking",
             fontsize=14, backgroundcolor='tab:red', color='white')

    # Iteration
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"Iteration $k = $" + f"{k}",
            fontsize=14)
    j += 1
    ax2.text(xpos, ypos - j * delypos,
             f"Step size " + f"$\\alpha_{{{k}}} = $" + f"{ak:.3f}",
             fontsize=14)
    
    # # Minimum point
    # j += 1
    # _xmin = f"{np.array2string(funclass.xmin.T[0], precision=3, floatmode='fixed')}"
    # ax2.text(xpos, ypos - j * delypos, f"$\\mathbf{{x}}^*$ = " + _xmin + r"$^\top$", fontsize=14)

    # # Current point
    # j += 1
    # _xk = f"{np.array2string(Xk[:, -1].T, precision=3, floatmode='fixed')}"
    # ax2.text(xpos, ypos - j * delypos, f"$\\mathbf{{x}}_{{{k}}}$ = " + _xk + r"$^\top$", fontsize=14)
    
    # # Current function value
    # j += 1
    # _fk = f"{funclass.func(Xk[0, -1], Xk[1, -1]):.3f}"
    # ax2.text(xpos, ypos - j * delypos, f"$f\\left(\\mathbf{{x}}_{{{k}}}\\right) = {_fk}$", fontsize=14)


def plot_contour():
    # Generate data for plotting
    x1 = np.linspace(-4, 4, 500)
    x2 = np.linspace(-4, 4, 500)
    X1, X2 = np.meshgrid(x1, x2)
    Z = funclass.func(X1, X2)

    # Plotting the contour
    if funclass.name == "Flipped Gaussian":
        ax.contour(X1, X2, Z, levels=np.logspace(-5, 5, 40),
                   cmap='Blues_r', linewidths=0.5)
    else:
        contours = ax.contour(X1, X2, Z, levels=np.logspace(-1, 5.5, 20),
                            cmap='Blues_r', linewidths=0.5)
    
    # Plot the three search directions
    _xk = Xk[:, -1].reshape(-1, 1)
    _grad = funclass.grad(_xk[0, 0], _xk[1, 0])
    _gradn = _grad / np.linalg.norm(_grad)
    _x, _y = get_search_lines(_xk[:, 0], _gradn[:, 0])
    
    # Current point
    ax.plot(_xk[0, 0], _xk[1, 0], 'tab:red', marker='o', markersize=6)
    ax.plot(Xk[0, :], Xk[1, :], 'tab:red', lw=1, alpha=0.8)
    ax.plot(_x, _y, 'black', lw=0.5, linestyle='--')
    ax.arrow(_xk[0, 0], _xk[1, 0], 0.5 * -_gradn[0, 0], 0.5 * -_gradn[1, 0],
             head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Plot backtracking points if any.
    if Xbt is not None:
        ax.plot(Xbt[0, :], Xbt[1, :], 'tab:gray', marker='o', markersize=4,
                linestyle="None")

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')
    
    # Plot the minimum point
    ax.plot(funclass.xmin[0], funclass.xmin[1], 'r*', markersize=10)

    # Set xlims and ylims
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Add labels and title
    ax.set_title(r"$f(\mathbf{x}) = $" + funclass.title, fontsize=18)

    # Update function plot along the search direction
    axins.cla()
    _dir ,_fdir = get_function_along_dir(_xk[:, 0], -_grad[:, 0])
    axins.plot(_dir[250], _fdir[250], 'tab:red', marker='o', markersize=3)
    axins.plot(_dir, _fdir, 'black', lw=1.0, alpha=0.7)

    # Find the position along the direction.
    if Xbt is not None:
        _t = [0.5 * ((_xb[0] - _xk[0, 0]) / -_grad[0, 0]
                     + (_xb[1] - _xk[1, 0]) / -_grad[1, 0])
              for _xb in Xbt.T]
        _f = [funclass.func(_xb[0], _xb[1]) for _xb in Xbt.T]
        axins.plot(_t, _f, 'tab:gray', marker='o', markersize=4, linestyle="None")

    # Remove top and right spines
    axins.spines['right'].set_visible(False)
    axins.spines['top'].set_visible(False)
    axins.spines.bottom.set_position(('axes', -0.01))    
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')

    # Set xlims and ylims
    axins.set_xlim(-10, 10)
    axins.set_ylim(np.min(Z),
                 np.max(Z))
    
    # Set title
    axins.set_title(r"$f(\mathbf{x} + t\mathbf{d})$", fontsize=18)


# Handling key press events
def on_press(event):
    global fig, funclass, methodclass, Xk, Xbt, ak, tau

    # Close figure if escaped.
    if event.key == 'escape':
        plt.close(fig)
        return

    # Choose which function to minimize.
    if event.key in function_dict.keys():
        funclass = function_dict[event.key]
        # Reset variables
        reset_params()
        ax.cla()
    
    # Return if no function has been selected.
    if funclass is None:
        return
    
    # Choose which mwthod to use for minimization.
    if event.key in method_dict.keys():
        methodclass = method_dict[event.key]
        # Reset variables
        reset_params()
        ax.cla()
    
    # Chekc if the solution needs to be updated.
    if event.key == 'right':
        # Compute the next step        
        # Update the solution
        _xk1 = update()
        if _xk1 is not None:
            Xk = np.hstack((Xk, _xk1))
        # Reset for backtracking
        Xbt = None
        # Clear axis
        ax.cla()
    elif event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Draw the plot and text
    update_text()
    plot_contour()
    # plot_dist_to_min()
    fig.canvas.draw()
    
    # Save plot
    if event.key == 'ctrl+s':
        fig.savefig("multivar_backtracking.png", dpi=300, bbox_inches='tight')
        fig.savefig("multivar_backtracking.pdf", bbox_inches='tight')


if __name__ == "__main__":
    # Function dictionary
    function_dict = {
        '0': aladaopt.Circle(xmin=np.array([2, 1])),
        '1': aladaopt.Ellipse(xmin=np.array([1, 0.5]),
                            Q=np.array([[3, 1], [1, 2]])),
        '2': aladaopt.Rosenbrock(a=1, b=5),
        '3': aladaopt.Quartic(xmin=np.array([2, 1]), a=2, b=5, c=3),
        '4': aladaopt.FlippedGaussian(xmin=np.array([0, 0]),
                                      Q=np.linalg.inv(np.array([[5, 2], [2, 3]])))
    }

    # Method dictionary
    method_dict = {
        'ctrl+0': aladaopt.GradientDescent,
        'ctrl+1': aladaopt.NewtonRaphson,
        'ctrl+2': aladaopt.LevenbergMarquardt
    }

    # Function and Method ID
    funclass = None
    methodclass = None

    # Create the figure and the axis.
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax.axis('off')
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="80%", height="40%", loc=4, borderpad=1)
    axins.axis('off')

    # Initialize the solution
    Xk = None
    Xbt = None
    ak = 5.0
    k = 0
    tau = 0.99
    reset_params()

    # Create the figure and the axis.
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Backtracking')
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.tight_layout(pad=3)
    plt.show()