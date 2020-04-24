# See: https://www.imagemagick.org/Usage/transform/#evaluate
# Note that RGB values are normalized to 0-1 so $X = (# pixels)/255

find . -name "*.JPEG" | xargs -I {}  convert {} \
-channel R -evaluate subtract $1 \
-channel G -evaluate subtract $2 \
-channel B -evaluate subtract $3 \
{}
