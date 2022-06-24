# Disparate Lens Stitching
Code for Master Thesis: Disparate Lens Stitching

It was examined how increasing disparicy between images affects
the results of different feature-matchers for image stitching.
Due to the lack of data all experiments were performed with synthetically
transformed long focal-length images. The plots in the results/plots directory
show that the synthetic data obviously obeys affine image motion. Also 
the descritor does not have to much influence on the process as the images
are captured at the same time and without large rotations or view-point 
changes. However it is hard to find matching keypoints with large scale
dicrepancies and small space overlap. Overall SIFT performed best but largely
due to upsampling the initial image before extracting the scale-space volume,
which probably only works so good because the datat is artificially created
via downsampling images. Other than that some algorithms perform poorly others
pretty okay even for larger disparities.

--------------------------------------------------------------------------------

# Install:
Base conda environment is specified in dls_env.yml
To extract keypoints and descriptors with R2D2 or 
KeyNet use the respective environments r2d2_env.yml
or keynet_env.yml. To use SURF opencv-contrib has
to be built with NON_FREE flag which is not available
in conda (ie. use a local install of similar version).

--------------------------------------------------------------------------------

## Data
For most scripts the data has to be in the following 
directory structure:
- data
    |_ seq0
    .
    .
    |_ seqN
        |_ ratio1
        .
        .
        |_ ratioN
            |_ c_short.png ... base focal length image
               c_long.png  ... base*ratioN focal length image
               h_short2long.csv ... (optional) ground truth homography 

To use R2D2 or KeyNet the keypoints and descriptors have to be precomputed
and stored in the files c_[short,long]_[kpt,dsc].npy files in the same 
structure as the images.

--------------------------------------------------------------------------------

## Code
* src/
    * stitch_extracted.py ... can be used to run stitching scripts with all algorithms
    * stich_pair.py       ... stitch two images provided via commandline
    * stich_synthetic.py  ... generate synthetic long focal lenght image and stitch
    * generate_pgt.py     ... generate GT homography by labeling 4 corresponding points 
* scripts/
    * run_eval.py         ... runs evaluation over all feature-matchers and disparities
    * plot_eval.py        ... generate plots from evaluation files

