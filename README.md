# MRTS
realize multi-resolution thin plate spline basis function in python 

## Material

MRTS (multi-resolution thin plate spline) basis function that can generate basis function according to the control points and saptial location

More on [paper](https://arxiv.org/pdf/1504.05659.pdf?fbclid=IwAR3WLhl5T150W1mmjBK2PShIPXmnOpyCJQ9uQnY81AxDcd2GlW2tFzp0A6g)

![](https://github.com/DongDong-Zoez/MRTS/blob/dc001e079b5b0f7e3241232a934b0625f8f7262b/assets/mrts.jpg)

![](https://github.com/DongDong-Zoez/MRTS/blob/dc001e079b5b0f7e3241232a934b0625f8f7262b/assets/tps.jpg)

## Usage

```python
import os
git clone https://github.com/DongDong-Zoez/MRTS
os.chdir('MRTS/mrts')
python demo.py
```

### On google colab

```python
from google.colab import drive
drive.mount('/content/gdrive')

%cd '/content/gdrive/MyDrive'
!git clone https://github.com/DongDong-Zoez/MRTS 

%cd 'MRTS/mrts'
!python basis.py -g False -c 'ctp.csv' -l 'ctp.csv'
```

### Speed up

We use cupy (a python built-in CUDA programming) to speed up the matrix operation,
with a 10000x10000 matrix, we only need 1 min to compute MRTS basis.
Make sure to turn on your GPU and set argumnet ```-g True```

### TO DO

- [ ] The algorithm is based on dissimilarity measurement, and to make the code easy to read, we calculate dissimilarity matrix directly, however, it is not necessary to calculate the dissimilarity matrix, we can use iteration to calculate the distance and overcome the ''out of memory'' problem alternatively.
