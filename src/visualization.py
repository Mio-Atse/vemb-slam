import matplotlib.pyplot as plt
import numpy as np

class MatplotlibVisualizer:
    def __init__(self, enable_segmentation=False):
        plt.ion()
        dpi = 100
        width_inches = 1080 / dpi
        height_inches = 720 / dpi

        self.fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        self.ax = self.fig.add_subplot(121 if enable_segmentation else 111, projection='3d')
        
        self.enable_segmentation = enable_segmentation
        self.ax_segmented = None
        if self.enable_segmentation:
            self.ax_segmented = self.fig.add_subplot(122, projection='3d')
            self.ax_segmented.set_xlabel('X')
            self.ax_segmented.set_ylabel('Y')
            self.ax_segmented.set_zlabel('Z')
            self.ax_segmented.set_title('SLAM Visualization (Segmented)')

        self.points = None
        self.colors = None
        self.camera_positions = []
        self.is_running = True
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('SLAM Visualization (Unsegmented)' if enable_segmentation else 'SLAM Visualization')
        
        self.fig.show()
        plt.pause(0.1)

    def update(self, points, colors, camera_position):
        if not self.is_running:
            return
        try:
            self.ax.cla()
            if len(points) > 0:
                max_points = 5000
                if len(points) > max_points:
                    idx = np.random.choice(len(points), max_points, replace=False)
                    vis_points = points[idx]
                    vis_colors = colors[idx]
                else:
                    vis_points = points
                    vis_colors = colors
                
                self.ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                              c=vis_colors, s=1)
            
            if camera_position is not None:
                self.camera_positions.append(camera_position)
                positions = np.array(self.camera_positions)
                self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2)
                self.ax.plot([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 'r*', markersize=10)
            
            if len(points) > 0:
                self._adjust_plot_limits(self.ax, points)
            
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('SLAM Visualization (Unsegmented)' if self.enable_segmentation else 'SLAM Visualization')
            
            plt.draw()
            plt.pause(0.01)
        except Exception as e:
            print(f"Error updating visualization: {str(e)}")

    def update_segmented(self, points, colors):
        if not self.is_running or not self.enable_segmentation:
            return
        try:
            self.ax_segmented.cla()
            if len(points) > 0:
                max_points = 5000
                if len(points) > max_points:
                    idx = np.random.choice(len(points), max_points, replace=False)
                    vis_points = points[idx]
                    vis_colors = colors[idx]
                else:
                    vis_points = points
                    vis_colors = colors
                
                self.ax_segmented.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], 
                                        c=vis_colors, s=1)
            
            if len(points) > 0:
                self._adjust_plot_limits(self.ax_segmented, points)
            
            self.ax_segmented.set_xlabel('X')
            self.ax_segmented.set_ylabel('Y')
            self.ax_segmented.set_zlabel('Z')
            self.ax_segmented.set_title('SLAM Visualization (Segmented)')
            
            plt.draw()
            plt.pause(0.01)
        except Exception as e:
            print(f"Error updating segmented visualization: {str(e)}")

    def _adjust_plot_limits(self, ax, points):
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                            points[:, 1].max() - points[:, 1].min(),
                            points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def close(self):
        plt.close(self.fig)
        self.is_running = False
