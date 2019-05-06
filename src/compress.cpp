
// opencv stuff
#include <iostream>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/contrib/contrib.hpp>

#include "openCVHelpers.h"


#include <sys/stat.h>

using namespace vdff;
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

//    /* 8bit, color or not */
//        cv::IMREAD_UNCHANGED  =-1,
//    /* 8bit, gray */
//        cv::IMREAD_GRAYSCALE  =0,
//    /* ?, color */
//        cv::IMREAD_COLOR      =1,
//    /* any depth, ? */
//        cv::IMREAD_ANYDEPTH   =2,
//    /* ?, any color */
//        cv::IMREAD_ANYCOLOR   =4

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
    if(images.size()>0){
        int status = mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        cout<<"mkdir status: "<<status<<endl;
    }

    for(int i=0; i<images.size(); i+=step){
        cv::Mat image;
        if(unchanged){
            image = cv::imread(images[i],cv::IMREAD_UNCHANGED);
        }else{
            image = cv::imread(images[i],(cv::IMREAD_COLOR & color*255 ) | (cv::IMREAD_ANYDEPTH & anydepth*255) | (cv::IMREAD_ANYCOLOR & anycolor*255) );
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


        std::vector<int> params;

        //http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#imwrite
        //Only 8-bit (or 16-bit unsigned (CV_16U) in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with ‘BGR’ channel order) images can be saved using this function.
        if(type == "jpg"){
           params.push_back(cv::IMWRITE_JPEG_QUALITY);
           params.push_back(compr);   // that's percent, so 100 == no compression, 1 == full
           int maxRange = MAX_RANGE(image);
           if(maxRange>255){
               double scaleFactor = 255.0f/maxRange; // this is 255/65234-1 or is it 1/256 fro 16->8 bits??
               image.convertTo(image,CV_8U,scaleFactor); //jpg only supports 8 bit,
           }
        }else if (type == "png"){
           params.push_back(cv::IMWRITE_PNG_COMPRESSION);
           params.push_back(compr);   // that's compression level, 9 == full , 0 == none
        }else if (type == "exr"){
            //exr can save 32 bit float
            openCVHelpers::convertToFloat(image);
        }else{
            cout<<"unsupported type"<<endl;
            exit(1);
        }


        if(debug) cv::imshow("out",image);
        cout<<"out: ";
        openCVHelpers::imgInfo(image);
        cout<<"\t path: "<<outname<<endl;

        //imwrite needs 0->255 or 0->256*256-1 (jpg 8bit, png 8/16bit)   or 0.0->1.0 float (exr 32 bit)
        cv::imwrite(outname,image,params);

        if(debug){
            image = cv::imread(outname,cv::IMREAD_UNCHANGED);
            cv::imshow("saved",image);
            cout<<"saved: ";
            openCVHelpers::imgInfo(image);
            cout<<"\t path: "<<outname<<endl;
            cv::waitKey();
        }

    }
}
