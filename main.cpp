#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert> 
#include <string> 
#include <cmath>
#include <chrono>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "brute_kernel.h"

typedef struct sites_type sites_t;

sites_t* init_sites(const int nsites, const int nrows, const int ncols)
{
    sites_t* sites = new sites_t[nsites];

    for (int i = 0; i < nsites; ++i)
    {
        sites[i].x = rand() % ncols;
        sites[i].y = rand() % nrows;

        // set color for given site
        sites[i].color.x = (sites[i].x*sites[i].y + 27 % 255);
        sites[i].color.y = (sites[i].x*sites[i].y*sites[i].x + 55) % 255;
        sites[i].color.z = (sites[i].x*sites[i].y*sites[i].y + 111) % 255;
        sites[i].color.w = 255;
    }

    return sites;
}

float calcDistance(const int x1, const int y1, const int x2, const int y2) 
{
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

void serialVoronoi(uchar4 *imrgba, sites_t* sites, const int numRows, const int numCols, const int numSites) {
    for (int y = 0; y < numRows; y++) 
    {
        for (int x = 0; x < numCols; x++) 
        {
            float min_dist = 999999;
            for (int i = 0; i < numSites; i++) 
            {
                float new_dist = calcDistance(x, y, sites[i].x, sites[i].y);
                    
                if (new_dist < min_dist) 
                {
                    min_dist = new_dist;
                    imrgba[y * numCols + x] = sites[i].color;
                }
            }
        }
    }
}

void enlarge_sites(uchar4* img, sites_t* sites, const int numRows, const int numCols, const int numSites)
{
    for (int i = 0; i < numSites; ++i)
    {
        uchar4 black = make_uchar4(0, 0, 0, 255);

        for (int r = -2; r <= 2; ++r)
        {
            for (int c = -2; c <= 2; ++c)
            {
                if (sites[i].y+r < 0 || sites[i].y+r >= numRows || 
                        sites[i].x+c < 0 || sites[i].x+c >= numCols)
                {
                    continue;
                }
                img[(sites[i].y+r)*numCols + (sites[i].x+c)] = black;   
            }
        }
    }
}

bool compare_uchar4(uchar4 first, uchar4 second)
{
    if (first.x != second.x)
    {
        return false;
    }
    else if (first.y != second.y)
    {
        return false;
    }
    else if (first.z != second.z)
    {
        return false;
    }
    else if (first.w != second.w)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool compare_imgs(uchar4* img1, uchar4* img2, const int nrows, const int ncols)
{
    int npixels = nrows*ncols;

    for (int i = 0; i < npixels; ++i)
    {
        if (!compare_uchar4(img1[i], img2[i]))
        {
            printf("%d\n", i);
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv) 
{
    using namespace std::chrono;

    srand(40698);

    int nsites, nrows, ncols;
    sites_t* h_sites;
    uchar4 *h_out_pix;

    sites_t* d_sites;
    uchar4* d_out_pix;

    uchar4* d2h_out_pix;

    if (argc != 4)
    {
        printf("ERROR: run as ./{program} {#_sites} {#rows} {#cols}\n");
        exit(0);
    }
    nsites = std::atoi(argv[1]);
    nrows = std::atoi(argv[2]);
    ncols = std::atoi(argv[3]);

    h_out_pix = new uchar4[nrows*ncols];
    d2h_out_pix = new uchar4[nrows*ncols];

    h_sites = init_sites(nsites, nrows, ncols);

    // serial shiznit
    auto start = high_resolution_clock::now();
    serialVoronoi(h_out_pix, h_sites, nrows, ncols, nsites);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    std::cout << "serial duration was: " << duration.count() << "ms" << std::endl;

    enlarge_sites(h_out_pix, h_sites, nrows, ncols, nsites);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_sites, nsites*sizeof(sites_t)));
    checkCudaErrors(cudaMalloc((void**)&d_out_pix, nrows*ncols*sizeof(uchar4)));

    // transfer to device
    checkCudaErrors(cudaMemcpy(d_sites, h_sites, nsites*sizeof(sites_t), cudaMemcpyHostToDevice));

    // call kernel
    launch_kernel(d_sites, d_out_pix, nsites, nrows, ncols);
    
    // transfer to host
    checkCudaErrors(cudaMemcpy(d2h_out_pix, d_out_pix, nrows*ncols*sizeof(uchar4), cudaMemcpyDeviceToHost));

    cv::Mat h_out_img(nrows, ncols, CV_8UC4, (void*)h_out_pix);
    if (!cv::imwrite("h_out.png", h_out_img))
    {
        printf("imwrite failed\n");
        exit(0);
    }

    cv::Mat d_out_img(nrows, ncols, CV_8UC4, (void*)d2h_out_pix);
    if(!cv::imwrite("d_out.png", d_out_img))
    {
        printf("imwrite failed\n");
        exit(0);
    }

    if (!compare_imgs(h_out_pix, d2h_out_pix, nrows, ncols))
    {
        printf("the images differ\n");
        exit(0);
    }

    // cleanup
    cudaFree(d_sites);
    cudaFree(d_out_pix);

    delete[] h_sites;
    delete[] h_out_pix;
    delete[] d2h_out_pix;

    return 0;
}



