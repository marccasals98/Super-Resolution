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