#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <cstdio> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <cmath> 
#include <math.h>
#include "utils.h"
#include "gaussian_kernel.h"
#include <chrono> 
using namespace std::chrono;

/* 
 * Compute if the two images are correctly 
 * computed. The reference image can 
 * either be produced by a software or by 
 * your own serial implementation.
 * */
void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems){

    for(int i = 0; i < numElems; i++){
        if(ref[i] - gpu[i] > 1e-5){
            std::cerr << "Error at position " << i << "\n"; 

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] <<"\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}



void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img; 

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);


    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows*ref_img.cols*ref_img.channels());
    std::cout << "PASSED!\n";


}

void gaussian_blur_filter(float *arr, const int f_sz, const float f_sigma=0.2){ 
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel 

    for(int r = -f_sz/2; r <= f_sz/2; r++){
        for(int c = -f_sz/2; c <= f_sz/2; c++){
            float fSum = expf(-(float)(r*r + c*c)/(2*f_sigma*f_sigma)); 
            arr[(r+f_sz/2)*f_sz + (c + f_sz/2)] = fSum; 
            filterSum  += fSum;
        }
    } 

    norm_const = 1.f/filterSum; 

    for(int r = -f_sz/2; r <= f_sz/2; ++r){
        for(int c = -f_sz/2; c <= f_sz/2; ++c){
            arr[(r+f_sz/2)*f_sz + (c + f_sz/2)] *= norm_const;
        }
    }
}



float calcDistance(const int x1, const int y1, const int x2, const int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0);
}



void serialVoronoi(uchar4 *imrgba, int *sites, const int numRows, const int numCols, const int numSites) {
    for (int y = 0; y < numRows; y++) {
        for (int x = 0; x < numCols; x++) {
            float dist = 9999999;
            for (int i = 0; i < numSites; i++) {
                if (dist < calcDistance(x, y, sites[2 * i], sites[2 * i + 1])) {
                    dist = calcDistance(x, y, sites[2 * i], sites[2 * i + 1]);
                    imrgba[y * numCols + x] = make_uchar4(i,i,i,255);
                }
            }
        }
    }
}


int main(int argc, char const *argv[]) {
   
    uchar4 *h_in_img, *h_o_img, *r_o_img; // pointers to the actual image input and output pointers  
    uchar4 *d_in_img, *d_o_img;
    int* sites;

    unsigned char *h_red, *h_blue, *h_green; 
    unsigned char *d_red, *d_blue, *d_green;   
    unsigned char *d_red_blurred, *d_green_blurred, *d_blue_blurred;   

    float *h_filter, *d_filter;  
    cv::Mat imrgba, o_img, r_img; 

    const int fWidth = 9; 
    const float fDev = 2;
    int numRows; 
    int numCols; 
    int numSites;


    switch(argc){
        //case 2:
          //  infile = std::string(argv[1]);
            //outfile = "blurred_gpu.png";
            //reference = "blurred_serial.png";
            //break; 
        //case 3:
          //  infile = std::string(argv[1]);
           // outfile = std::string(argv[2]);
           // reference = "blurred_serial.png";
           // break;
        case 4:
            numRows = std::atoi(argv[1]);
            numCols = std::atoi(argv[2]);
            numSites = std::atoi(argv[3]);
            break;
        default: 
                std::cerr << "Usage ./gblur <Number of Rows> <Number of Columns> <Number of Voronoi Sites> \n";
                exit(1);

   }

    // preprocess 
    //cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR); 
    //if(img.empty()){
      //  std::cerr << "Image file couldn't be read, exiting\n"; 
        //exit(1);
    //}
    imrgba.create(numRows, numCols, CV_8UC4);
    //cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);

    o_img.create(numRows, numCols, CV_8UC4);
    r_img.create(numRows, numCols, CV_8UC4);
    
    //const int numRows = img.rows;
    //const int numCols = img.cols;
    const size_t  numPixels = numCols * numRows;
    //const int numSites = numPixels / 10;
    sites = (int*)malloc(2 * numSites * sizeof(int));
    for (int i = 0; i < numSites; i++) {
        sites[i * 2] = rand() % (numCols + 1);
        sites[(i * 2) + 1] = rand() % (numRows + 1);
    }
    
    
    h_in_img = imrgba.ptr<uchar4>(0); // pointer to input image 
    h_o_img = o_img.ptr<uchar4>(0); // pointer to output image 
    r_o_img = r_img.ptr<uchar4>(0); // pointer to reference output image 

    // allocate the memories for the device pointers  
    
    checkCudaErrors(cudaMalloc((void**)&d_in_img, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_red, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_green, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_blue, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_red_blurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_green_blurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_blue_blurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_o_img, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float) * fWidth * fWidth));

    // filter allocation 
    h_filter = new float[fWidth*fWidth];
    gaussian_blur_filter(h_filter, fWidth, fDev); // create a filter of 9x9 with std_dev = 0.2  

    printArray<float>(h_filter, 81); // printUtility.

    // copy the image and filter over to GPU here 
    checkCudaErrors(cudaMemcpy(d_in_img, h_in_img, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * fWidth * fWidth, cudaMemcpyHostToDevice));

    // kernel launch code 
    your_gauss_blur(d_in_img, d_o_img, numRows, numCols, d_red, d_green, d_blue, 
            d_red_blurred, d_green_blurred, d_blue_blurred, d_filter, fWidth);


    // memcpy the output image to the host side.
    checkCudaErrors(cudaMemcpy(h_o_img, d_o_img, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost));


    // perform serial memory allocation and function calls, final output should be stored in *r_o_img
    //  ** there are many ways to perform timing in c++ such as std::chrono **
    h_red = (unsigned char*)malloc(numPixels);
    h_green = (unsigned char*)malloc(numPixels);
    h_blue = (unsigned char*)malloc(numPixels);
    
    unsigned char* h_red_blurred = new unsigned char[numPixels];
    unsigned char* h_green_blurred = new unsigned char[numPixels];
    unsigned char* h_blue_blurred = new unsigned char[numPixels];

    auto start = high_resolution_clock::now();
    serialVoronoi(h_in_img, sites, numRows, numCols, numSites);
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("The execution time in microseconds for serial implementation: ");
    std::cout << duration;
    printf("\t");



    // create the image with the output data 
    //cv::Mat output(numRows, numCols, CV_8UC4, (void*)h_o_img); // generate GPU output image.
    //bool suc = cv::imwrite(outfile.c_str(), output);
    //if(!suc){
     //   std::cerr << "Couldn't write GPU image!\n";
      //  exit(1);
    //}
    cv::Mat output_s(numRows, numCols, CV_8UC4, (void*)h_in_img); // generate serial output image.
    suc = cv::imwrite("Voronoi_serial.png", output_s);
    if(!suc){
        std::cerr << "Couldn't write serial image!\n";
        exit(1);
    }


    // check if the caclulation was correct to a degree of tolerance

    //checkResult(reference, outfile, 1e-5);

    // free any necessary memory.
    cudaFree(d_in_img);
    cudaFree(d_o_img);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    cudaFree(d_red_blurred);
    cudaFree(d_green_blurred);
    cudaFree(d_blue_blurred);
    delete [] h_filter;
    return 0;
}



