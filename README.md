# mview3dpose
 multiple view multiple user 3d pose estimation
 
 ## Instalation
 
 ```
 # 1. Create a conda virtual environment.
conda create -n mview3dpose python=3.6
conda activate mview3dpose

# 2. Install PyTorch
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch

# 3. Get Repository
https://github.com/AntonioEscamilla/mview3dpose.git
cd mview3dpose

# 4. install
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install -r requirements.txt
```
