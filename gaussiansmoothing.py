import numpy as np

def compute_smooth_value(i,x,y,n):
    startidx = max(0,i-n)
    endidx = min(i+n, len(x)-1)
    kernel_sum = 0
    sigma = 1
    val = 0
    for j in range(startidx, endidx+1):
        diff = np.abs(x[j] - x[i])
        kernel_at_pos = np.exp(-diff*diff / (2 * sigma * sigma))
        val += kernel_at_pos*y[j]
        kernel_sum += kernel_at_pos
    val = val/kernel_sum
    return val


def gaussian_kernel_smoothing(x, y, n):
    y_smooth = [compute_smooth_value(i,x,y,n) for i in range(len(x))]
    return y_smooth
