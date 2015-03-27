
// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "openCVHelpers.h"

#include <iostream>

using namespace vdff;
using namespace std;
using namespace Utils;

int main(int argc, char **argv) {
    std::cout<<"hello"<<std::endl;

    string indir="../samples/sim";
    getParam("indir", indir, argc, argv);

    string outdir="outdir";
    getParam("outdir", outdir, argc, argv);

    int step=1;
    getParam("step", step, argc, argv);

    int compr=90;
    getParam("compr", compr, argc, argv);

    string type="jpg";
    getParam("type", type, argc, argv);

    if(type != "jpg" && type != "png"){
        cout<<"wront type"<<endl;
        exit(1);
    }

//    /* 8bit, color or not */
//        CV_LOAD_IMAGE_UNCHANGED  =-1,
//    /* 8bit, gray */
//        CV_LOAD_IMAGE_GRAYSCALE  =0,
//    /* ?, color */
//        CV_LOAD_IMAGE_COLOR      =1,
//    /* any depth, ? */
//        CV_LOAD_IMAGE_ANYDEPTH   =2,
//    /* ?, any color */
//        CV_LOAD_IMAGE_ANYCOLOR   =4

    bool color = 0;
    bool anydepth = 0;
    bool anycolor = 0;
    bool unchanged = 0;

    getParam("unchanged", unchanged, argc, argv);
    //grayscale if no parameter set
    getParam("color", color, argc, argv);
    getParam("anydepth", anydepth, argc, argv);
    getParam("anycolor", anycolor, argc, argv);

    bool debug = true;
    getParam("debug",debug,argc,argv);



    vector<string> images = Utils::getAllImagesFromFolder(indir.c_str());

    for(int i=0; i<images.size(); i+=step){
        cv::Mat image;
        if(unchanged){
            image = cv::imread(images[i],CV_LOAD_IMAGE_UNCHANGED);
        }else{
            image = cv::imread(images[i], (CV_LOAD_IMAGE_COLOR & color*255 ) | (CV_LOAD_IMAGE_ANYDEPTH & anydepth*255) | (CV_LOAD_IMAGE_ANYCOLOR & anycolor*255) );
        }

        if(debug) cv::imshow("in",image);
        cout<<"in: ";
        openCVHelpers::imgInfo(image);
        cout<<"\t path: "<<images[i]<<endl;


        stringstream ss;
        string filename = images[i].substr(indir.size());
        std::size_t pos = filename.find(".");
        std::string name = filename.substr(0,pos);
        ss<<outdir<<getOSSeparator()<<name<<"."<<type;
        string outname=ss.str();

        openCVHelpers::convertToFloat(image);
        if(debug) cv::imshow("out",image);
        cout<<"out: ";
        openCVHelpers::imgInfo(image);
        cout<<"\t path: "<<outname<<endl;


        std::vector<int> params;

        //http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#imwrite
        //Only 8-bit (or 16-bit unsigned (CV_16U) in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with ‘BGR’ channel order) images can be saved using this function.
        if(type == "jpg"){
           int maxRange = MAX_RANGE(image);
           if(maxRange>255){
               double scaleFactor = 255/maxRange; // this is 255/65234-1 or is it 1/256 fro 16->8 bits??
               image.convertTo(image,CV_8U,scaleFactor); //jpg only supports 8 bit,
           }
           params.push_back(CV_IMWRITE_JPEG_QUALITY);
           params.push_back(compr);   // that's percent, so 100 == no compression, 1 == full
        }else{
           params.push_back(CV_IMWRITE_PNG_COMPRESSION);
           params.push_back(compr);   // that's compression level, 9 == full , 0 == none
        }

        //TODO jpeg-2000:
        //Durch die Wavelet-Transformation vermeidet JPEG 2000 im Gegensatz zu JPEG-1 störende Blockartefakte bei hoher Kompression. Stattdessen tendieren die Bilder zu Unschärfeartefakten sowie Schatten an harten Kontrasten. JPEG 2000 eignet sich besonders für große Bilder, da hier die größeren Blöcke bei Waveletfiltern Vorteile gegenüber den recht kleinen 8×8-Blöcken der DCT von JPEG-1 haben. Bei kleineren Bildern kann, je nach Bildinhalt, auch JPEG-1 einen Qualitätsvorteil bieten.
        //imwrite needs 0->255 or 0->256*256-1
        cv::imwrite(outname,image,params);

        if(debug){
            image = cv::imread(outname,CV_LOAD_IMAGE_UNCHANGED);
            cv::imshow("saved",image);
            cout<<"saved: ";
            openCVHelpers::imgInfo(image);
            cout<<"\t path: "<<outname<<endl;
            cv::waitKey();
        }

    }
}
