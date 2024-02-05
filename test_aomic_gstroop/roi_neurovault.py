from nilearn import image
from nilearn import plotting
from nilearn.datasets import load_sample_motor_activation_image

full_map = load_sample_motor_activation_image()
thresholded_map = image.threshold_img(full_map, threshold=5)
plotting.plot_roi(
    thresholded_map,
    display_mode="z",
    output_file="thresholded_map_neurovault.png",
)

binary_map = image.binarize_img(thresholded_map)

binary_map.to_filename(f"roi_neurovault.nii.gz")
plotting.plot_roi(
    binary_map, output_file="roi_neurovault.png", display_mode="z"
)
