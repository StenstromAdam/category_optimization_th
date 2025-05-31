import matplotlib.pyplot as plt

def print_sphere(points, path, categories = 0, colors = 0):
    '''
        Creates a 2 or 3-dimensional plot from the given points to the specified path.
        If specified as a list, category names are also printed to their corresponding point on the plot.
    '''

    dim = len(points[0])

    if dim not in (2, 3):
        print("Can only sphere plot for 2 and 3 dimensions. Currently ", dim, "\n No plot generated...")
        return
        


    plt.style.use('ggplot')
    if dim == 2:
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1, color='c', fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)
        if hasattr(colors, 'any') and colors.any() != 0:
            ax.scatter(points[:,0], points[:,1], c=colors[:len(points[:,0])])
        else:
            ax.scatter(points[:,0], points[:,1])
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        if categories != 0:
            for i, label in enumerate(categories):
                ax.annotate(label, (points[i, 0], points[i, 1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=6)
    
    
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        if hasattr(colors, 'any') and colors.any() != 0:
            ax.scatter(points[:,0], points[:,1], points[:,2], c=colors[:len(points[:,0])])
        else:
            ax.scatter(points[:,0], points[:,1], points[:,2])
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        if categories != 0:
            for i, label in enumerate(categories):
                ax.text(points[i,0], points[i,1], points[i,2], label, ha='center', fontsize=6)
    
    ax.set_aspect('equal')  # Ensure the plot is square
    plt.savefig(path, format='png')