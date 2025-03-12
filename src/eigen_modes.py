import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

def test_eigenmode():
    c=1
    N = 20
    NEIGHBORS = [-N, -1, 1, N]
    A = np.zeros([N**2, N**2])
    for i in range(N**2):
        A[i, i] = -4
        for di in NEIGHBORS:
            if not 0<= i+di < N**2:
                continue
            print(i, di, i%N)
            if i%N==0 and di ==-1:
                continue
            elif i%N==N-1 and di==1:
                continue
            else:                
                A[i, i+di] = 1

    plt.imshow(A)
    plt.colorbar()
    plt.show()
    
    w, vl, vr = scipy.linalg.eig(A, left=True, right=True)
    w = w.astype(np.float64)
    vr = vr.astype(np.float64)
    vl = vl.astype(np.float64)
    print(w)
    # plt.imshow(vr)
    # plt.colorbar()
    # plt.show()

    
    # for mode in range(N**2-10, N**2):
    for mode in range(N**2):
    # grid = w.reshape([N,N])
        grid = vl[:,mode].reshape([N,N])
        plt.imshow(grid)
        plt.colorbar()
        plt.title(w[mode])
        plt.show()

if __name__ == '__main__':
    test_eigenmode()