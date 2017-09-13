"""
A very simple little script to fit a set of data with (optionall) 
some noise with different polynomial orders
"""
import numpy as np
import matplotlib.pyplot as plt


def test_func(x):
    return 1 - x*x + 0.444444*x - 0.1*x**3 + np.sin(x*2*np.pi)

def generate_mock_data(N=20,  seed=100, noise=None):
    """
    Generate the mock data to use. 
    """

    # Ensure we have a fixed set of data if seed is set
    if seed is not None:
        np.random.seed(seed)


    x_min = -1
    x_max = 2.
    x = np.random.uniform(x_min, x_max, size=N)
#    x = np.array([0., 0.45, 0.46, 0.48])
    y = test_func(x)

    if noise is not None:
        dy = np.random.normal(0, scale=noise, size=len(y))
        y = y+dy
    
    return x, y


def MSE(y, y_pred, dy=None):
    """
    Mean squared error error function
    """

    if dy is not None:
        diff = np.sum(((y-y_pred)/dy)**2)/len(y)
    else:
        diff = np.sum(((y-y_pred))**2)/len(y)
        
    return diff


def predict(x, res):
    p = np.poly1d(res)
    return p(x)
    
def show_fit(x, y, res):
    p = np.poly1d(res)

    xgrid = np.linspace(np.min(x), np.max(x), 200)
    y_pred = p(xgrid)

    plt.plot(xgrid, test_func(xgrid), color='gray')
    plt.scatter(x, y)
    plt.plot(xgrid, y_pred, color='red')
    plt.show()


def calculate_mse_curve(x, y):
    """
    Calculate a MSE curve for power 1 to 10
    """

    alpha = range(1, 11)
    mses = np.zeros(len(alpha))

    for i, a in enumerate(alpha):
        res = np.polyfit(x, y, a)
        y_pred = predict(x, res)

        mses[i] = MSE(y, y_pred)
        

    return alpha, mses


def show_mse_grid_linear():

    # Just some simple linear regression

    np.random.seed(101)
    x = np.random.uniform(-1, 1, size=10)
    true_slope = 1.3
    y = x*true_slope + np.random.normal(0, 0.3, size=len(x))
    xplot = np.linspace(-1.5, 1.5, 100)
    
    slopes = np.linspace(0, 1.8, 100)
    mses = [MSE(y, x*b) for b in slopes]
    mses = np.array(mses)

    mse_true = MSE(y, x*true_slope)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(x, y)
    ax[0].plot(xplot, xplot*true_slope, color='red')
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[1].plot(slopes, mses)
    ax[1].scatter(true_slope, mse_true)
    ax[1].set_xlabel("slope")
    ax[1].set_ylabel("MSE")

    fig.set_size_inches(12,4)
    fig.savefig("Figures/mse_grid.pdf")
    
