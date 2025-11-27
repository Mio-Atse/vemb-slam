import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def visualize_trajectory(file_path):
    # Read the data from the file
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    except IOError:
        print(f"Error: Unable to read file '{file_path}'. Please make sure the file exists and is readable.")
        return
    except ValueError:
        print(f"Error: Invalid data format in '{file_path}'. Please ensure the file contains comma-separated X, Y, Z coordinates.")
        return

    # Extract X, Y, and Z coordinates
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(X, Y, Z, label='Camera Trajectory')

    # Add markers for start and end points
    ax.scatter(X[0], Y[0], Z[0], color='green', s=100, label='Start')
    ax.scatter(X[-1], Y[-1], Z[-1], color='red', s=100, label='End')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory in 3D')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize 3D camera trajectory.')
    parser.add_argument('file_path', type=str, help='Path to the input file containing trajectory data')
    
    # Get the file path from user input
    args = parser.parse_args()
    
    # Call the visualization function with the provided file path
    visualize_trajectory(args.file_path)
