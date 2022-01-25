# Super-Resolution
## FSRCNN (Fast Super Resolution Convolutional Neural Network)
![alt text](https://github.com/marccasals98/Super-Resolution/blob/main/results/RESULTS_ART/result_bar2.png)
![alt text](https://github.com/marccasals98/Super-Resolution/blob/main/results/RESULTS_ART/gat.png)

Implementation of the FSRCNN made by Dong et al. in this [article](https://arxiv.org/pdf/1608.00367.pdf). The names of the different Networks are displayed with the name of the dataset that they were trained with.

### MNIST

This network was trained with MNIST dataset, so the super-resolution works only with handwritten numbers.

![alt text](https://github.com/marccasals98/Super-Resolution/blob/main/results/MNIST/figura_3.png)

### BSDS300

This network uses a general purpous dataset. It only has 200 images for trainning and 100 for test. So the result could be better.

### BSDS300_DA

This network uses the same dataset as before but using Data Augemntation.

### Flickr 

This network uses a 31k images dataset. It has an input shape of (150,150,3) and an output shape of (300,300,3).
The results using this network are the best of all project.

![alt text](https://github.com/marccasals98/Super-Resolution/blob/main/results/FLICKR/avio_final.png)

### Flickr

This network uses the same dataset as the previous one but with some changes in the complexity of the Network that increases the results for doing an upscaling of x4.
The images shown at the beginning are done using this network.
![alt text](https://github.com/marccasals98/Super-Resolution/blob/main/results/RESULTS_ART/papallona2.png)

## Structure of the Network

Following the structure described in the article of Chao Dong et al:

![alt text](https://github.com/marccasals98/Super-Resolution/blob/main/results/RESULTS_ART/structure%20(1).PNG)

We add some modifications:

- Because we are trainning colour images we need 3 channels in the input (except in the MNIST).
- We use the Adam optimizer instead of the SGD (Stochastic Gradient Descent).
- In flickr x4 we add some network complexity by turning the m=4 into m=7.



## References 

CHOLLET FRANÇOIS, Deep Learning with Python, 2018. 

CHAO DONG, CHEN CHANGE LOY, XIAOOU TANG Accelerating the Super-Resolution Convolutional Neural Network, 2016. https://arxiv.org/pdf/1412.6980.pdf

P. KINGMA DIEDERIK, LEI BA JIMMY, Adam: A method for stochastic optimization, 2014. https://arxiv.org/pdf/1412.6980.pdf

CHAO DONG, CHEN CHANGE LOY, KAIMING HE, XIAOOU TANG, Image Super-Resolution Using Deep Convolutional Networks, 2015. https://arxiv.org/pdf/1501.00092.pdf

GOODFELLOW IAN, BENGIO YOSHUA, COURVILLE AARON, Deep Learning

NIELSEN MICHAEL, Neural Networks and Deep Learning, 2019.



