from flask import Flask, render_template
from PIL import Image
import numpy as np
import math
import scipy.fftpack as fftpack

app = Flask(__name__)

@app.route('/')
def index():
    # Open the grayscale image
    img = Image.open('bird.png').convert('L')
    # Convert image to ndarray
    im1 = np.asarray(img)
    # Performing FFT
    c = np.abs(fftpack.fft2(im1))
    # Shifting the Fourier frequency image
    d = fftpack.fftshift(c)
    e = d.astype(float)

    M = d.shape[0]
    N = d.shape[1]
    # H is defined and values in H are initialized to 1
    H1 = np.ones((M,N))
    H2 = np.ones((M,N))

    center1 = M/2
    center2 = N/2
    d_0 = 30.0
    t1 = 1
    t2 = 2 * t1

    # Defining the convolution function for BLPF
    for i in range(1,M):
        for j in range(1,N):
            r1 = (i - center1) ** 2 + (j - center2) ** 2
            # Euclidean distance from origin is computed
            r = math.sqrt(r1)
            # Using cut-off radius to eliminate high frequency
            if r > d_0:
                H1[i,j] = 1/(1 + (r/d_0)**t1)
                H2[i,j] = 1/(1 + (r/d_0)**t2)

    # H is converted from ndarray to image
    H1 = H1.astype(float)
    H1 = Image.fromarray(H1)
    H2 = H2.astype(float)
    H2 = Image.fromarray(H2)

    # Performing the convolution
    con1 = d * H1
    con2 = d * H2

    # Computing the magnitude of the inverse FFT
    z1 = np.abs(fftpack.ifft2(con1))
    z2 = np.abs(fftpack.ifft2(con2))

    # Converting the images from ndarray to image
    im1 = Image.fromarray(e.astype(np.uint8))
    im2 = Image.fromarray(z1.astype(np.uint8))
    im3 = Image.fromarray(z2.astype(np.uint8))

    # Save the images
    im1.save("im1.png")
    im2.save("im2.png")
    im3.save("im3.png")

    return render_template('index.html', im1='im1.png', im2='im2.png', im3='im3.png')

if __name__ == '__main__':
    app.run()