# image-clustering

This is a simple image clustering algorithm which uses KMeans for clustering and performs 3 types of vectorization of images using vgg16, vgg19 and resnet50 using the weights from ImageNet

output folder will be created and the different clusters formed using he different algorithms will be present. 

change the following variables(present in the main() function) for your convinience:
1) number_of_clusters  
2) data_path - this is the path of the folder that contans the different images that we will pass to the algorithm.

Python Modules used: Keras.io, tensorflow(as backend for keras), os, time, random, cv2(openCV), numpy, sklearn(scikit-learn), shutil, glob

To run: "python image_clustering.py"

