"""
Script for create an animation to demonstrate how the line search direction 
impacts the optimization process.

Author: Sivakumar Balasubramanian
Date: 14 March 2024
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
    global funclass, xk, theta
    # Reset the solution
    xk = np.random.rand(2) * 8 - 4
    theta = np.array([0, 45, -45])


def get_function_along_dir(dirang):
    _t = np.linspace(-10, 10, 501)
    _x = xk[0] + _t * np.cos(np.deg2rad(dirang))
    _y = xk[1] + _t * np.sin(np.deg2rad(dirang))
    return _t, funclass.func(_x, _y)


def get_search_lines():
    global theta, xk
    _x = [xk[0] + 10 * np.array([-np.cos(_t), np.cos(_t)])
          for _t in np.deg2rad(theta)]
    _y = [xk[1] + 10 * np.array([-np.sin(_t), np.sin(_t)])
          for _t in np.deg2rad(theta)]
    return _x, _y


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
    _x, _y = get_search_lines()
    ax.plot(xk[0], xk[1], 'tab:red', marker='o', markersize=10)
    ax.plot(_x[0], _y[0], 'black', lw=0.5, linestyle='--')
    ax.plot(_x[1], _y[1], 'red', lw=0.5, linestyle='--')
    ax.plot(_x[2], _y[2], 'green', lw=0.5, linestyle='--')
    ax.arrow(xk[0], xk[1], np.cos(np.deg2rad(theta[0])), np.sin(np.deg2rad(theta[0])),
             head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(xk[0], xk[1], np.cos(np.deg2rad(theta[1])), np.sin(np.deg2rad(theta[1])),
             head_width=0.05, head_length=0.1, fc='red', ec='red')
    ax.arrow(xk[0], xk[1], np.cos(np.deg2rad(theta[2])), np.sin(np.deg2rad(theta[2])),
             head_width=0.05, head_length=0.1, fc='green', ec='green')
    
    # Plot the minimum point
    ax.plot(funclass.xmin[0], funclass.xmin[1], 'r*', markersize=10)

    # Remove top and  right spines
    ax.axis('off')

    # Set xlims and ylims
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Add labels and title
    ax.set_title(r"$f(\mathbf{x}) = $" + funclass.title, fontsize=18)

    # Update function plot along the search direction
    ax2.cla()
    _dir ,_fdir = get_function_along_dir(theta[0])
    ax2.plot(_dir[250], _fdir[250], 'tab:red', marker='o', markersize=3)
    ax2.plot(_dir, _fdir, 'black', lw=1.0, alpha=0.7)
    _dir ,_fdir = get_function_along_dir(theta[1])
    ax2.plot(_dir, _fdir, 'red', lw=1.0, alpha=0.7)
    _dir ,_fdir = get_function_along_dir(theta[2])
    ax2.plot(_dir, _fdir, 'green', lw=1.0, alpha=0.7)

    # Remove top and right spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines.bottom.set_position(('axes', -0.01))    
    ax2.spines['bottom'].set_color('#bbbbbb')
    ax2.spines['left'].set_color('#bbbbbb')

    # Set xlims and ylims
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(np.min(Z),
                 np.max(Z))
    
    # Set title
    ax2.set_title(r"$f(\mathbf{x} + t\mathbf{d})$", fontsize=18)


# Handling key press events
def on_press(event):
    global funclass, xk, theta

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
    

    # Check if the step size needs to be updated.
    if event.key == 'up':
        theta += 5
    elif event.key == 'down':
        theta -= 5
    # Clear axis
    ax.cla()
    
    # Reset the plot
    if event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Draw the plot and text
    plot_contour()
    fig.canvas.draw()
    
    # Save plot
    if event.key == 'ctrl+s':
        fig.savefig("multivar_linesearch_demo.png", dpi=300, bbox_inches='tight')
        fig.savefig("multivar_linesearch_demo.pdf", bbox_inches='tight')


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

    # Initialize the solution
    theta = np.array(np.array([0, 45, -45]))
    xk = None
    reset_params()

    # Create the figure and the axis.
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Line Search')
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.tight_layout(pad=3)
    plt.show()

# """
# Script for create an animation to demonstrate how the line search direction 
# impacts the optimization process.

# Author: Sivakumar Balasubramanian
# Date: 14 March 2024
# """

# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import aladaopt

# mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
# mpl.rcParams['toolbar'] = 'None' 


# def reset_params():
#     global funclass, xk, theta
#     # Reset the solution
#     xk = np.random.rand(2) * 8 - 4
#     theta = 10


# def get_function_along_dir(xk, dirvec):
#     _t = np.linspace(-10, 10, 501)
#     _x = xk[0] + _t * dirvec[0]
#     _y = xk[1] + _t * dirvec[1]
#     return _t, funclass.func(_x, _y)


# def get_function_along_angle(xk, dirang):
#     _t = np.linspace(-10, 10, 501)
#     _x = xk[0] + _t * np.cos(np.deg2rad(-dirang))
#     _y = xk[1] + _t * np.sin(np.deg2rad(-dirang))
#     return _t, funclass.func(_x, _y)


# def get_search_line_along_dir(xk, dirvec):
#     _x = xk[0] + 10 * np.array([-dirvec[0], dirvec[0]])
#     _y = xk[1] + 10 * np.array([-dirvec[1], dirvec[1]])
#     return _x, _y


# def get_search_line():
#     global theta, xk
#     _t = -np.deg2rad(theta)
#     _x = xk[0] + 10 * np.array([-np.cos(_t), np.cos(_t)])
#     _y = xk[1] + 10 * np.array([-np.sin(_t), np.sin(_t)])
#     return _x, _y


# def plot_contour():
#     # Generate data for plotting
#     x1 = np.linspace(-4, 4, 500)
#     x2 = np.linspace(-4, 4, 500)
#     X1, X2 = np.meshgrid(x1, x2)
#     Z = funclass.func(X1, X2)

#     # Plotting the contour
#     if funclass.name == "Flipped Gaussian":
#         ax.contour(X1, X2, Z, levels=np.logspace(-5, 5, 40),
#                    cmap='Blues_r', linewidths=0.5)
#     else:
#         contours = ax.contour(X1, X2, Z, levels=np.logspace(-1, 5.5, 20),
#                             cmap='Blues_r', linewidths=0.5)
    
#     # Plot the three search directions
#     _x, _y = get_search_line()
#     ax.plot(xk[0], xk[1], 'tab:red', marker='o', markersize=10)
#     ax.plot(_x[0], _y[0], 'tab:red', lw=0.5, linestyle='--')
#     ax.arrow(xk[0], xk[1], np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)),
#              head_width=0.05, head_length=0.1, fc='tab:red', ec='tab:red')
    
#     # Plot the three search directions
#     _grad = funclass.grad(xk[0], xk[1])
#     _gradn = _grad / np.linalg.norm(_grad)
#     _x, _y = get_search_line_along_dir(xk, _gradn[:, 0])
    
#     # Current point
#     ax.plot(_x, _y, 'black', lw=0.5, linestyle='--')
#     ax.arrow(xk[0], xk[1], -_gradn[0, 0], -_gradn[1, 0],
#              head_width=0.05, head_length=0.1, fc='black', ec='black')
    
#     # Plot the minimum point
#     ax.plot(funclass.xmin[0], funclass.xmin[1], 'r*', markersize=10)

#     # Remove top and  right spines
#     ax.axis('off')

#     # Set xlims and ylims
#     ax.set_xlim(-4, 4)
#     ax.set_ylim(-4, 4)

#     # Add labels and title
#     ax.set_title(r"$f(\mathbf{x}) = $" + funclass.title, fontsize=18)

#     # Update function plot along the search direction
#     ax2.cla()
#     _dir ,_fdir = get_function_along_dir(xk, _gradn[:, 0])
#     ax2.plot(_dir[250], _fdir[250], 'tab:red', marker='o', markersize=3)
#     ax2.plot(_dir, _fdir, 'black', lw=1.0, alpha=0.7)
#     _dir ,_fdir = get_function_along_angle(xk, theta)
#     ax2.plot(_dir, _fdir, 'tab:red', lw=1.0, alpha=0.7)

#     # Remove top and right spines
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines.bottom.set_position(('axes', -0.01))    
#     ax2.spines['bottom'].set_color('#bbbbbb')
#     ax2.spines['left'].set_color('#bbbbbb')

#     # Set xlims and ylims
#     ax2.set_xlim(-10, 10)
#     ax2.set_ylim(np.min(Z),
#                  np.max(Z))
    
#     # Set title
#     ax2.set_title(r"$f(\mathbf{x} + t\mathbf{d})$", fontsize=18)


# # Handling key press events
# def on_press(event):
#     global funclass, xk, theta

#     # Close figure if escaped.
#     if event.key == 'escape':
#         plt.close(fig)
#         return

#     # Choose which function to minimize.
#     if event.key in function_dict.keys():
#         funclass = function_dict[event.key]
#         # Reset variables
#         reset_params()
#         ax.cla()
    
#     # Return if no function has been selected.
#     if funclass is None:
#         return
    

#     # Check if the step size needs to be updated.
#     if event.key == 'up':
#         theta += 5
#     elif event.key == 'down':
#         theta -= 5
#     # Clear axis
#     ax.cla()
    
#     # Reset the plot
#     if event.key in ['r', 'R']:
#         # Reset the plot
#         reset_params()
#         ax.cla()
    
#     # Draw the plot and text
#     plot_contour()
#     fig.canvas.draw()
    
#     # Save plot
#     if event.key == 'ctrl+s':
#         fig.savefig("multivar_linesearch_demo.png", dpi=300, bbox_inches='tight')
#         fig.savefig("multivar_linesearch_demo.pdf", bbox_inches='tight')


# if __name__ == "__main__":
#     # Function dictionary
#     function_dict = {
#         '0': aladaopt.Circle(xmin=np.array([2, 1])),
#         '1': aladaopt.Ellipse(xmin=np.array([1, 0.5]),
#                             Q=np.array([[3, 1], [1, 2]])),
#         '2': aladaopt.Rosenbrock(a=1, b=5),
#         '3': aladaopt.Quartic(xmin=np.array([2, 1]), a=2, b=5, c=3),
#         '4': aladaopt.FlippedGaussian(xmin=np.array([0, 0]),
#                                       Q=np.linalg.inv(np.array([[5, 2], [2, 3]])))
#     }

#     # Method dictionary
#     method_dict = {
#         'ctrl+0': aladaopt.GradientDescent,
#         'ctrl+1': aladaopt.NewtonRaphson,
#         'ctrl+2': aladaopt.LevenbergMarquardt
#     }

#     # Function and Method ID
#     funclass = None
#     methodclass = None

#     # Create the figure and the axis.
#     fig = plt.figure(figsize=(15, 8))
#     gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
#     ax = fig.add_subplot(gs[0, 0])
#     ax.equal_aspect = True
#     ax.axis('off')
#     ax2 = fig.add_subplot(gs[0, 2])
#     ax2.axis('off')

#     # Initialize the solution
#     theta = np.array(np.array([0, 45, -45]))
#     xk = None
#     reset_params()

#     # Create the figure and the axis.
#     fig.canvas.manager.set_window_title('ALADA Optimization Animations: Line Search')
#     fig.canvas.mpl_connect('key_press_event', on_press)
#     plt.tight_layout(pad=3)
#     plt.show()