The predict.py and train.py have been configured according to instructions in the Berrijam Jam 2024 Submission Template.

The code is intended to use tensorflow with a GPU. As according to the documentation in:
https://www.tensorflow.org/install/source#gpu

CUDA 12.2 is only compatible with cuDNN 8.9 and tensorflow 2.15.0, which are the versions we tested with. We have added code in install.sh to install CUDA 12.2 and cuDNN 8.9 on an Ubuntu machine, and add relevant files to the path so python will be able to reference the installation. 

Please ensure that you use 'source 1_install.sh' command for installing the dependencies because we set environment variables (CUDA_DIR, PATH, and LD_LIBRARY_PATH) within the script.

A requirements.txt file has been provided with the python packages for running our model. If possible, please ensure these steps run successfully. 

Pre-trained models have been downloaded and referenced in the resources directory

The code assumes that if the output path is multi-level, the folders before the final level have already been initialised beforehand.

EDA and Model selection have been performed in a notebook, submitted under resources/EDA

The Licenses have been saved under resources/LICENSES directory.

The style_content.png file under resources/pretrained is an art work, Composition VII (1913) by Wassily Kandinsky, which is under the public domain.