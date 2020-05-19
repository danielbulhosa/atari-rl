# Instructions

Download Imagenet 2012 training and validation datasets from
http://academictorrents.com/collection/imagenet-lsvrc-2015

You should get two tars:
ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar

Run trainprep.sh and valprep.sh from this directory with
these two tars in the directory. The scripts will create
Training and Validation directories, unzip the tar files,
and organize the images into folders corresponding to
the Imagenet classes.

A common preprocessing step is to resize the Imagenet 
images to size 256x256. Running preprocess_images.sh
will do this. 

The file sysnet_words.txt maps class codes to class 
names. The Python scripts in this repository contain
code for mapping class codes to class one-hot encodings
and class names. 

Finally, the subtract rgb mean script does exactly what
it says. However, we do NOT use this script in our work
since we do the mean subtraction in the generator along
with the augmentations (in Python). It's just here for
convenience if needed to process in bulk.
