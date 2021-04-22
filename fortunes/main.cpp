#include <iostream>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "Voronoi.h"
#include "VPoint.h"

vor::Voronoi * v;
vor::Vertices * ver; // vrcholy
vor::Vertices * dir; // smìry, kterými se pohybují
vor::Edges * edg;	 // hrany diagramu

void create_output_img(const char* name, const int nrows, const int ncols)
{
    cv::Mat output(nrows, ncols, CV_8UC1, cv::Scalar(255));

    for (vor::Vertices::iterator i = ver->begin(); i != ver->end(); ++i)
    {
        cv::Point pt((*i)->x, (*i)->y);

        cv::circle(output, pt, 3, cv::Scalar(0), cv::FILLED, cv::LINE_8, 0);
    }

    for (vor::Edges::iterator i = edg->begin(); i != edg->end(); ++i)
    {
        cv::Point pt1((*i)->start->x, (*i)->start->y);
        cv::Point pt2((*i)->end->x, (*i)->end->y);

        cv::line(output, pt1, pt2, cv::Scalar(0), 1, cv::LINE_8, 0);
    }

    if (!cv::imwrite(name, output))
    {
        printf("imwrite failed\n");
        exit(0);
    }
}

int main (int argc, char **argv) 
{
	using namespace vor;
    using namespace std::chrono;

    int nsites, nrows, ncols;

	v = new Voronoi();
	ver = new Vertices();
	dir = new Vertices();

	srand (40698);

    if (argc != 4)
    {
        printf("ERROR: run as ./{program} {#_sites} {#rows} {#cols}\n");
        exit(0);
    }
    nsites = std::atoi(argv[1]);
    nrows = std::atoi(argv[2]);
    ncols = std::atoi(argv[3]);
    
	for(int i=0; i<nsites; i++) 
	{

		ver->push_back(new VPoint( ncols * (double)rand()/(double)RAND_MAX , nrows * (double)rand()/(double)RAND_MAX )); 
		dir->push_back(new VPoint( (double)rand()/(double)RAND_MAX - 0.5, (double)rand()/(double)RAND_MAX - 0.5)); 
	}

    auto start = high_resolution_clock::now();
	edg = v->GetEdges(ver, nrows, ncols);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
	std::cout << "voronois done!\n";
    std::cout << "serial duration was: " << duration.count() << "us" << std::endl;

	for(vor::Edges::iterator i = edg->begin(); i!= edg->end(); ++i)
	{
			if( (*i)->start == 0 )
			{
				std::cout << "chybi zacatek hrany!\n";
				continue;
			}
			if( (*i)->end == 0 )
			{
				std::cout << "chybi konec hrany!\n";
				continue;
			}	
	}
    
    create_output_img("h_fortune.png", nrows, ncols);

    return 0;
}
