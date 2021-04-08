#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"

/*
 * The launcher for your kernels. 
 * This is a single entry point and 
 * all arrays **MUST** be pre-allocated 
 * on device. you must implement all other 
 * kernels in the respective files.
 * */ 

struct sites_type
{
    int x, y;
    uchar4 color;
};

void launch_kernel(struct sites_type* sites, uchar4* out_pix, const int nsites, const int nrows, const int ncols);
