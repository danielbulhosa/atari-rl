import os
import cv2
import pickle
import cupy as cp
import time
import shared.definitions.paths as paths

root_dir = paths.training
class_dirs = [class_dir for class_dir in os.listdir(root_dir)
              if class_dir[-4:] != ".tar"]
assert len(class_dirs) == 1000, """Not exactly 1000 classes were found!
                                   {} classes were found instead 
                                """.format(len(class_dirs))

num_pixels = 0
sum_cov = cp.array([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]])

with open('pixel_avg.pkl', 'rb') as file:
    results = pickle.load(file)

pixel_avg = results['pixel_avg']

t_job_start = time.time()
delta_ts = []

# Would be good to refactor this and the above into a couple
# of methods
for num, class_dir in enumerate(class_dirs):
    t_dir_start = time.time()

    full_class_path = root_dir + class_dir + '/'
    class_images = [image for image in os.listdir(full_class_path)
                    if image[-5:] == ".JPEG"]

    for image in class_images:
        image_path = full_class_path + image
        pixel_array = cp.array(cv2.imread(image_path)).reshape(-1, 3)

        # Mean needs to be subtracted for PCA
        sum_cov += cp.matmul((pixel_array - pixel_avg).T,
                             pixel_array - pixel_avg)
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
    'sum_cov': sum_cov,
    'num_pixels': num_pixels,
    'cov': sum_cov / (num_pixels - 1)
}

with open('cov.pkl', 'wb') as file:
    pickle.dump(results, file)