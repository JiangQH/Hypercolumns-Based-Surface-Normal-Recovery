import matplotlib.pyplot as plt

def show_filter(data, iter):
    plt.figure(iter)
    filt_min, filt_max = data.min(), net.blobs['conv'].data.max()