

# Screening:

## Build Singularity Images:


'''
source envs.sh
make build_images
make build_base_sif
'''


## Use locally only for developments:

After build all images:

'''
source envs.sh
make run
'''

Inside of the image:

'''
make
source activate.sh
'''

When everything its ready for production, commit everything and build once again the images.

'''
make build_images
make build_prod_sif
'''

## Production:

'''
singularity exec /path/to/screening_prod.sif ...
'''
