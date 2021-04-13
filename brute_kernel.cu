#include "./brute_kernel.h"

#define BLOCK 16

typedef struct sites_type sites_t;

__device__
float calc_dist(int x1, int x2, int y1, int y2)
{
    return sqrtf(powf(x1-x2,2) + powf(y1-y2,2));
}

__global__
void brute_kernel(sites_t* sites, uchar4* out_pix, const int nsites, const int nrows, const int ncols)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

//    uchar4 black = make_uchar4(0, 0, 0, 255);

    if (y < nrows && x < ncols)
    {
        int index = y*ncols + x;
        
        float min_dist = 99999999;
        int min_dist_site = -1;

        for (int s = 0; s < nsites; ++s)
        {
            float current_dist = calc_dist(x, sites[s].x, y, sites[s].y);

            /*if (current_dist == min_dist)
            {
                min_dist_site = -1;
                out_pix[index] = black;
            }*/
            /*else*/ if (current_dist < min_dist)
            {
                min_dist = current_dist;
                min_dist_site = s;
            }
        }

        //if (min_dist_site != -1)
        //{
            out_pix[index] = sites[min_dist_site].color;
    
        //}
        
    }
}

__global__
void s_brute_kernel(const sites_t* sites, uchar4* out_pix, const int nsites, const int nrows, const int ncols)
{
    const int shared_size = 32;

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    uchar4 my_color;
    __shared__ sites_t s_sites[shared_size];

    if (y < nrows && x < ncols)
    {
        int global_index = y*ncols + x;
        int local_index = threadIdx.y*blockDim.x + threadIdx.x;

        float min_dist = 99999999;
        
        for (int s = 0; s < nsites; s += shared_size)
        {
            // load sites into shared memory
            if (local_index < shared_size && local_index+s < nsites)
            {
                s_sites[local_index] = sites[local_index+s];
            }

            __syncthreads();

            for (int i = 0; i < shared_size; ++i)
            {
                // only run for valid sites
                if (s+i >= nsites)
                {
                    break;
                }

                float current_dist = calc_dist(x, s_sites[i].x, y, s_sites[i].y);

                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    my_color = s_sites[i].color;
                }
            }

            __syncthreads();
        }

        out_pix[global_index] = my_color;
    }
}

// color the sites of the diagram black
__global__
void color_sites(sites_t* sites, uchar4* out_pix, const int nsites, const int nrows, const int ncols)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < nsites)
    {
        uchar4 black = make_uchar4(0, 0, 0, 255);

        for (int r = -2; r <= 2; ++r)
        {
            for (int c = -2; c <= 2; ++c)
            {
                // make the sites appear bigger than 1 pixel
                if (sites[i].y + r < 0 || sites[i].y + r >= nrows ||
                        sites[i].x + c < 0 || sites[i].x + c >= ncols)
                {
                    continue;
                }

                out_pix[(sites[i].y+r)*ncols + (sites[i].x+c)] = black;
            }
        }
    }
}

void launch_kernel(sites_t* sites, uchar4* out_pix, const int nsites, const int nrows, const int ncols)
{
    dim3 block_dim(BLOCK,BLOCK,1);
    dim3 grid_dim((ncols-1)/BLOCK+1,(nrows-1)/BLOCK+1,1);
 
    s_brute_kernel<<<grid_dim, block_dim>>>(sites, out_pix, nsites, nrows, ncols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    color_sites<<<nsites/256+1, 256>>>(sites, out_pix, nsites, nrows, ncols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
