# Hate speech


Solution was based on Polbert model from hugginface transformers. As the dataset size was relatively small in order to avoid overfiting to particular words embedding module of BERT was frozen during the training process.

The training data is stored in the repository and it is assumed that it is static and not mutable - in real world the training process should be also dockerized and possibly dynamic data should have been kept in some kind of remote storage eg. AWS S3.

The code for the whole project could be easily written in more generic way - for example it is preassumed for bert classifier that there are 3 possible classes. Also mapping from classnames to class ids has been hardcoded. In the context of this particular problem it is (at last for me) more readable.

The whole training procedure with definition of configuration files, printout of validation / test results is held and described within notebook `train.ipynb`

It is preassumed that training is held on machine with GPU (and Cuda 11.1 installed), there were 2 reasons for that:
- training on cpu is extremely slow
- my laziness - didnt want to write this all if statements to check the device and allocating tensors


To run training:
- Create python3.6 virtualenvironment by running `virtualenv -p python3.6 myenv`
- Activate the enivroment `source myenv\bin\activate`
- Install the gpu reqirements `pip install -r requirements_gpu.txt -f https://download.pytorch.org/whl/torch_stable.html`
- Install the package `pip install -e.`


Deployment procedure is documented and described within README file in deployment diretory. 
Deployment is supported only for CPU from the reasons similar to supporting training only on GPU + pytorch cpu version is much smaller though much smaller docker image could be produced.