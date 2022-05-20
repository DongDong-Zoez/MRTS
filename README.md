# MRTS
realize multi-resolution thin plate spline basis function in python 


## Usage

```python
git clone https://github.com/DongDong-Zoez/MRTS
python demo.py
```

### On colab



```python
from google.colab import drive
drive.mount('/content/gdrive')

import os 
os.chdir('/content/gdrive/MyDrive/MRTS/mrts')

!python basis.py -g True -c 'ctp.csv' -l 'ctp.csv'
```

## Material

MRTS (multi-resolution thin plate spline) basis function that can generate basis function according to the control points and saptial location

More on [paper](https://arxiv.org/pdf/1504.05659.pdf?fbclid=IwAR3WLhl5T150W1mmjBK2PShIPXmnOpyCJQ9uQnY81AxDcd2GlW2tFzp0A6g)

![](https://github.com/DongDong-Zoez/MRTS/blob/dc001e079b5b0f7e3241232a934b0625f8f7262b/assets/mrts.jpg)

![](https://github.com/DongDong-Zoez/MRTS/blob/dc001e079b5b0f7e3241232a934b0625f8f7262b/assets/tps.jpg)
