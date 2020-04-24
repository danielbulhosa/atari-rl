# The find command finds all files whose name matches the expression.
# find . looks in the current directory, and the -name flag looks for name matches.
# (See https://www.lifewire.com/uses-of-linux-command-find-2201100 for more) 

# The xargs command is used to apply the proceeding operation to the preceeding args
# (See https://www.howtoforge.com/tutorial/linux-xargs-command/ for more)

# convert is a utility that allows for resizing images. 
# - The -resize flag specifies we want to resize the image.
# - The 256 number tells the command we want to resize to 256x256
# - The > symbol tells convert only to shrink to size if image is larger, not to enlarge to fit
# - The ^ symbol tells convert to shrink the image so that the shortest side fits the correct dimensio (this is what is done for Alexnet!)
# - The {} throughout are placeholders for the files output by find. The last {} tells the command to rewrite the files to their same name.
# - -gravity Center tells convert to crop the image from the center
# - 256X256+0+0 tells convert to crop down to 256x256, with 0 offset from the center
# - +repage tells convert to shrink the Virtual Canvas to fit the new image (rather than the original one)
# For more see:
# - Cropping: https://imagemagick.org/Usage/crop/
# - Resizing: https://imagemagick.org/Usage/resize/
# - Idea for this command: https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-practical-cnns-in-pytorch-1-61daa679c74a
# Note that we can chain convert commands, that was our idea!

# Also note find is recursive so we can use it from this directory

find . -name "*.JPEG" | xargs -I {}  convert {} -resize 256x256^\> -gravity Center -crop 256x256+0+0 +repage {}
