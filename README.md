# image-clustering

This is a simple unsupervised image clustering algorithm which uses KMeans for clustering and performs 3 types of vectorization of images using vgg16, vgg19 and resnet50 using the weights from ImageNet

A folder named "output" will be created and the different clusters formed using the different algorithms will be present. 

Change the following two variables(present in the main() function) for your convinience:
1) number_of_clusters  
2) data_path - This is the path of the folder that contans the different images that we will pass to the algorithm.

Python Modules used: 
-Keras.io
-tensorflow(as backend for keras)
-os
-time
-random
-cv2(openCV)
-numpy
-sklearn(scikit-learn)
-shutil
-glob

To run: "python image_clustering.py"

