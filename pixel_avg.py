import os
import cv2
import pickle
import cupy as cp
import time

# Number of class folders
os.system('ls datasets/ILSVRC2012/Training | wc -l')

root_dir = "datasets/ILSVRC2012/Training/"
class_dirs = [class_dir for class_dir in os.listdir(root_dir)
              if class_dir[-4:] != ".tar"]
assert len(class_dirs) == 1000, """Not exactly 1000 classes were found!
                                   {} classes were found instead 
                                """.format(len(class_dirs))

num_pixels = 0
sum_pixels = cp.array([0.0, 0.0, 0.0])

t_job_start = time.time()
delta_ts = []

# Using the GPU shaves 30% (10 minutes) from the total calculation!
# Could be more if im.read created a cp.array directly?
# Doesn't seem supported though, by OpenCV
for num, class_dir in enumerate(class_dirs):
    t_dir_start = time.time()

    full_class_path = root_dir + class_dir + '/'
    class_images = [image for image in os.listdir(full_class_path)
                    if image[-5:] == ".JPEG"]

    for image in class_images:
        image_path = full_class_path + image
        pixel_array = cp.array(cv2.imread(image_path)).reshape(-1, 3)

        sum_pixels += pixel_array.sum(axis=0)
        num_pixels += pixel_array.shape[0]

    t_dir_end = time.time()
    delta_t = (t_dir_end - t_dir_start) / 60
    delta_ts.append(delta_t)
    avg_delta_t = round(sum(delta_ts) / len(delta_ts), 2)

    print("""
    Iteration #{} Completed in {} minutes.
    Average iteration time: {} minutes
    Estimated time remaining: {} minutes
    """.format(num + 1,
               round(delta_t, 2),
               avg_delta_t,
               avg_delta_t * (1000 - (num - 1))
               ))

results = {
    'sum_pixels': sum_pixels,
    'num_pixels': num_pixels,
    'pixel_avg': sum_pixels / num_pixels
}

with open('pixel_avg.pkl', 'wb') as file:
    pickle.dump(results, file)