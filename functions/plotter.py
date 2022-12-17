import matplotlib.pyplot as plt

# Plot the scattered values of the data (x,y)
def plotter(data, label=['train']):
    
    for d,l in zip(data, label):
        x, y = d
        plt.scatter(x, y, label=l)
    
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    
    plt.show()

# Plot the behavior of loss over all iterations
def loss_plotter(max_epoch, loss):
    x = list(range(max_epoch))
    plt.plot(x,loss)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss")
    
    plt.show()