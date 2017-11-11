# Convolution_Neutral_Networks
'Transfer Learning' was used to classify the images into the 40 categories. Inception-v3, a CNN model that was pre-trained by Google on 100K images with 1000 categories was used in the process.

******************************************************************************************************************************

As the retrained_graph.pb file was 88 MB, due to size restrictions, I have uploaded them in the folllowing URL.
The One-Drive folder also consists of Bottlenecks that were created during the predictions

https://mcgill-my.sharepoint.com/personal/nirmal_kanagasabai_mail_mcgill_ca/_layouts/15/guestaccess.aspx?docid=19b10f61652704239a18eecd259fe3eba&authkey=Ad7ExX0ggXwPQ9hR5vDoyBc

******************************************************************************************************************************


*To ensure the Convolution Neural Networks program run, the following steps are to be followed:

1) Docker Installation
2) Tensorflow Installation
3) Docker Optimization
4) Image Extraction
5) Directory Structure Creation
6) Download Inceptionv3
7) Build Tensorflow (Retrain the model)
8) Make Predictions*

------------------------------------------------------------------------------------------------------------------------------

#Skipping the installation process

------------------------------------------------------------------------------------------------------------------------------

#Docker Optimization:

- To cater to the computational needs of the program, it is mandatory to tweak the parameters as the speed of the processor in the virtual machine is crucial.

- As our project involves a lot of number crunching, we had to optimize the configuration for speed.

The following steps were carried out:

[1] VirtualBox (ACPI Shutdown)
[2] Base Memory increased to 75% of Laptop's total Memory
[3] Processor increased to the maximum applicable
[4] Headless Start

------------------------------------------------------------------------------------------------------------------------------

#Image Extraction:

Please refer to 'Image_Extraction.ipynb' file

------------------------------------------------------------------------------------------------------------------------------

#Creating Tensorflow Docker Image:

docker run -it gcr.io/tensorflow/tensorflow:latest-devel

------------------------------------------------------------------------------------------------------------------------------

#Directory Structure Creation:

It is very important to ensure that the directory structure is obeyed when we are adopting Transfer learning approach.

tf_files
   |_ _ _ Train_Image (Directory containing sub-directories)
   |_ _ _ Test_Image (Directory containing all test images)
   |_ _ _ Bottlenecks (which will be created later)
   |_ _ _ Retrained_Labels.txt (which will be created later)
   |_ _ _ Retrained_Graph.pb (which will be created later)
   |_ _ _ Label_Image.py (The prediction file)
   |_ _ _ Inception (which will be downloaded from Github)

Train_Image
   |_ _ _ 0
   |_ _ _ 1
   |_ _ _ 2
   |_ _ _ 3
   |_ _ _ .
   |_ _ _ .
   |_ _ _ .
   |_ _ _ 39

0,1,2,3,..,39 represents the 40 classes of Tiny Imagenet Challenge

------------------------------------------------------------------------------------------------------------------------------

#Sharing the folders with the VM:

docker run -it -v $HOME/tf_files:/tf_files gcr.io/tensorflow/tensorflow:latest-devel

------------------------------------------------------------------------------------------------------------------------------

#Downloading Inception V3:

Allow us to retrain the Inception V3 classifier with our Tiny Imagenet dataset.

git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git pull origin master
git checkout 6d46c0b370836698a3195a6d73398f15fa44bcb2

------------------------------------------------------------------------------------------------------------------------------

#Building the Code:

bazel build -c opt --copt=-mavx tensorflow/examples/image_retraining:retrain

------------------------------------------------------------------------------------------------------------------------------

#Running the Tensorflow Retraining process:

-> Bottleneck directory will be used to cache the outputs of the lower layers on disk so that they do not have to repeatedly be recalculated

-> Retraned_Graph.pb will be the retrained model for the Tiny Imagenet classifier

-> Retrained_Labels.txt will contain the list of all the classes that is involved and that we will use to correspond the test images to

bazel-bin/tensorflow/examples/image_retraining/retrain \
--bottleneck_dir=/tf_files/bottlenecks \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/Train_Image

------------------------------------------------------------------------------------------------------------------------------

#Prediction:

Label_Image.py

------------------------------------------------------------------------------------------------------------------------------
