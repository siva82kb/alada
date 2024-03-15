"""
Script for create an animation to demonstrate the Gradient descent method with 
a fixed step size.

Author: Sivakumar Balasubramanian
Date: 04 March 2024
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import aladaopt

mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
mpl.rcParams['toolbar'] = 'None' 


def reset_params():
    global funclass, Xk, ak, mu
    # Reset the solution
    Xk = np.random.rand(2, 1) * 8 - 4
    ak = 0.025
    mu = 1


def update():
    _xk = Xk[:, -1].reshape(-1, 1)
    _grad = funclass.grad(_xk[0, 0], _xk[1, 0])
    if methodclass.name == "Gradient Descent":
        return methodclass.update(_xk, ak, _grad)

    # Methods using the Hessian
    _hess = funclass.hessian(_xk[0, 0], _xk[1, 0])
    if methodclass.name == "Newton-Raphson":
        return methodclass.update(_xk, _grad, _hess)
    elif methodclass.name == "Levenberg-Marquardt":
        return methodclass.update(_xk, _grad, _hess, mu)

    return None


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
    
    # Plot the search trajectory
    ax.plot(Xk[0, :], Xk[1, :], 'tab:red', lw=1.0)
    
    # Plot the minimum point
    ax.plot(funclass.xmin[0], funclass.xmin[1], 'r*', markersize=10)
    ax.plot(Xk[0, -1], Xk[1, -1], 'ko', markersize=5)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.spines['left'].set_color('#bbbbbb')

    # Set xlims and ylims
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Add labels and title
    ax.set_title(r"$f(\mathbf{x}) = $" + funclass.title, fontsize=18)


def plot_dist_to_min():
    axins.cla()
    # Update the inset plot.
    _k = Xk.shape[1]
    _dist = [np.linalg.norm(_x - funclass.xmin.T[0]) for _x in Xk.T]
    axins.plot(np.arange(_k), _dist, lw=1, color="tab:green")
    axins.set_xlim(0, (_k // 40 + 1) * 40)
    axins.set_ylim(np.min([-5, 1.1 * np.min(_dist)]),
                   np.max([5, 1.1 * np.max(_dist)]),)
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
    ax2.text(xpos, ypos - j * delypos, f"Method: {methodclass.name}",
             fontsize=14, backgroundcolor='tab:red', color='white')

    # Iteration
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"Iteration $k = $" + f"{k}",
            fontsize=14)
    
    # Step size
    if methodclass.name == "Gradient Descent":
        j += 1
        ax2.text(xpos, ypos - j * delypos,
                 f"Step size " + f"$\\alpha_{{{k}}} = $" + f"{ak:.3f}",
                 fontsize=14)
    # Mu term
    if methodclass.name == "Levenberg-Marquardt":
        j += 1
        ax2.text(xpos, ypos - j * delypos,
                 f"$\\mu = $" + f"{mu:.3f}",
                 fontsize=14)
    
    # Minimum point
    j += 1
    _xmin = f"{np.array2string(funclass.xmin.T[0], precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos, f"$\\mathbf{{x}}^*$ = " + _xmin + r"$^\top$", fontsize=14)

    # Current point
    j += 1
    _xk = f"{np.array2string(Xk[:, -1].T, precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos, f"$\\mathbf{{x}}_{{{k}}}$ = " + _xk + r"$^\top$", fontsize=14)
    
    # Current function value
    j += 1
    _fk = f"{funclass.func(Xk[0, -1], Xk[1, -1]):.3f}"
    ax2.text(xpos, ypos - j * delypos, f"$f\\left(\\mathbf{{x}}_{{{k}}}\\right) = {_fk}$", fontsize=14)


# Handling key press events
def on_press(event):
    global funclass, methodclass, Xk, ak, mu

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
    
    # Return if no method has been selected.
    if methodclass is None:
        # Draw the plot and text
        plot_contour()
        fig.canvas.draw()
        return

    # Check if the step size needs to be updated.
    if event.key == 'up':
        if methodclass.name == "Gradient Descent":
            # Increase the step size
            ak = 1.05 * ak
        elif methodclass.name == "Levenberg-Marquardt":
            # Increase the mu term
            mu = 1.05 * mu
    elif event.key == 'down':
        if methodclass.name == "Gradient Descent":
            # Decrease the step size
            ak = np.max([0.001, 0.95 * ak])
        elif methodclass.name == "Levenberg-Marquardt":
            # Decrease the mu term
            mu = np.max([0.001, 0.95 * mu])
    
    # Chekc if the solution needs to be updated.
    if event.key == 'right':
        # Compute the next step        
        # Update the solution
        _xk1 = update()
        if _xk1 is not None:
            Xk = np.hstack((Xk, _xk1))
        # Clear axis
        ax.cla()
    elif event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Draw the plot and text
    update_text()
    plot_contour()
    plot_dist_to_min()
    fig.canvas.draw()


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
    fig = plt.figure(figsize=(12, 7.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax.axis('off')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    axins = inset_axes(ax2, width="80%", height="30%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Gradient Descent')

    # Initialize the solution
    ak, mu = 0, 0
    reset_params()

    # Create the figure and the axis.
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Gradient Descent')
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.tight_layout(pad=3)
    plt.show()