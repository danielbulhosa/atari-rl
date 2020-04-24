# Create a training directory and unzip master training tar file there
mkdir Training
mv ILSVRC2012_img_train.tar Training
cd Training 
tar -xvf ILSVRC2012_img_train.tar

# Loop through outputed training class tars (n*.tar), unzip, and put into their own folders
for file in n*.tar
do 
  echo Creating directory ${file%.*}...
  mkdir ${file%.*} 
  echo Moving $file into ${file%.*}...
  mv $file ${file%.*} 
  echo Entering directory ${file%.*}...
  cd ${file%.*} 
  echo Unziping tar file
  tar -xvf *.tar 
  echo Exiting directory ${file%.*} and back to parent directory... 
  cd .. 
done
