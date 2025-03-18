import numpy as np
import matplotlib.pyplot as plt

def problem_3a():
    # The original piecewise function
    def piecewise_function(x):
        return np.piecewise(x, 
                            [x < -0.1, 
                            (x >= -0.1) & (x < 3), 
                            (x >= 3) & (x < 5), 
                            x >= 5],
                            [lambda x: -x**3, 
                            lambda x: -3/100*x-1/500, 
                            lambda x: -(x-31/10)**3 - 23/250,
                            lambda x: 1083/200*(x-6) **2 - 6183/500])
    
    # Gradient descent
    def gradient_descent(x):
        x = np.clip(x, -1e6, 1e6) # Avoid overflow for better print outs
        return np.piecewise(x, 
                            [x < -0.1, 
                            (x >= -0.1) & (x < 3), 
                            (x >= 3) & (x < 5), 
                            x >= 5],
                            [lambda x: -3*x**2, 
                            lambda x: -3/100, 
                            lambda x: -3*(x-31/10)**2, 
                            lambda x: 1083/50 * (x - 6)])

    # Define the range of x values
    x = np.linspace(-4, 10, num=200) 
    plt.plot(x, piecewise_function(x), label=r'$f(x)$')
    plt.plot(x, gradient_descent(x), label=r'$f^{\prime}(x)$')

    learning_rates = [.001, .01, .1, 1, 10]  # Different learning rates (best as of now [.24003]
    colors = ['red', 'green', 'blue', 'orange', 'pink', 'yellow']    # Colors for the learning rates
    starting_x = -3  # Starting x value

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Problem 3a')

    for i, learning_rate in enumerate(learning_rates):
        x_value = starting_x

        x_points = []
        y_points = []
        
        for n in range(100):  # 100 iterations
            y_value = gradient_descent(x_value)
            
            # Only append points within -4 and 10
            if -4 <= x_value <= 10:
                x_points.append(x_value)
                y_points.append(y_value)
            
            # Calculate gradient and update x
            new_x = gradient_descent(x_value)
            x_value -= learning_rate * new_x
            
            # Print debug information
            if x_value > 4.5 and x_value < 10:
                if y_value < 5 and y_value > -5:
                    print(f"LR: {learning_rate}, Iteration: {n}, x_value: {x_value}, y_value: {y_value}, gradient: {new_x}")
        
        plt.scatter(x_points, y_points, color=colors[i], label=f'LR = {learning_rate}')

    plt.grid(True)
    plt.legend()
    plt.show()
    
def problem_3b():
    file_path = 'gradient_descent_sequence.txt'
    x1_values = []
    x2_values = []

    # Read and parse the file to set up scatterplot
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into two parts and convert to float
            x1, x2 = map(float, line.split())
            x1_values.append(x1)
            x2_values.append(x2)
    
    # Convert the lists to numpy arrays
    x1_values = np.array(x1_values)
    x2_values = np.array(x2_values)
    
    # Create a grid of points for the contour plot
    x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1_values) - 4, max(x1_values) + 4),
                                   np.linspace(min(x2_values) - 1, max(x2_values) + 1))

    # Function for contour        
    def f(x1, x2, a1, a2, c1, c2):
        return a1 * (x1 - c1)**2 + a2 * (x2 - c2)**2

    # Values for the function
    a1 = 1
    a2 = 1/3.75
    c1 = .4
    c2 = 4
    
    # Compute the function values on the grid
    f_values = f(x1_grid, x2_grid, a1, a2, c1, c2)

    # Plot the contour
    contour = plt.contour(x1_grid, x2_grid, f_values)
    plt.colorbar(contour, label='Function Value')
            
    # Plot the scatter plot
    plt.scatter(x1_values, x2_values, color='blue', marker='o', label='Problem 3b')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.title('Problem 3b')
    plt.legend()
    plt.axis('equal')
    plt.show()
    
    
def problem_3c():
    # The orginal function
    def f(x):
        return 2 / 3 * np.abs(x) ** (3 / 2)
    
    # Define the gradient function
    def gradient_function(x):
        return x/np.sqrt(np.abs(x))

    # Define the range of x values
    x = np.linspace(-10, 10)
    y = f(x)
    plt.plot(x, y, label=r'$f(x)$')
    y = gradient_function(x)
    plt.plot(x, y, label=r'$f^{\prime}(x)$')

    learning_rates = [.001, .01, .1]  # Different learning rates
    colors = ['red', 'green', 'blue']  # Colors for the scatter plots
    starting_x = 3  # Starting x value

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Problem 3c')

    for i, learning_rate in enumerate(learning_rates):  # iterate over learning rates
            x_value = starting_x

            x_points = []
            y_points = []
            iteration = 0
            
            # This was for us to see the behavior of the gradient descent when we did not give a set iteration
            while gradient_function(x_value) >= 0.05 or gradient_function(x_value) <= -0.05:  # breaking computer
                iteration += 1
                y_value = gradient_function(x_value)
                x_points.append(x_value)
                y_points.append(y_value)
                
                new_x = gradient_function(x_value)
                x_value -= learning_rate * new_x
                
            print("iterations for ",learning_rate," was ",iteration)
            plt.scatter(x_points, y_points, color=colors[i], label=f'LR = {learning_rate}')

    plt.grid(True)
    plt.legend()
    plt.show()
    
# When running close out a graph window to view next graph.
def run_all():
    problem_3a()
    problem_3b()
    problem_3c()

run_all()