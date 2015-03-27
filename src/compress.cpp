
// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "loading.h"

#include "openCVHelpers.h"

#include <iostream>


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



    vector< std::pair<string,string> > images = getAllImagesFromFolder2(indir.c_str());

    for(int i=0; i<images.size(); i+=step){
        cv::Mat image;
        if(unchanged){
            image = cv::imread(images[i].first,CV_LOAD_IMAGE_UNCHANGED);
        }else{
            image = cv::imread(images[i].first, (CV_LOAD_IMAGE_COLOR & color*255 ) | (CV_LOAD_IMAGE_ANYDEPTH & anydepth*255) | (CV_LOAD_IMAGE_ANYCOLOR & anycolor*255) );
        }

        stringstream ss;

        string& filename = images[i].second;
        std::size_t pos = filename.find(".");
        std::string name = filename.substr(0,pos);

        cout<<images[i].first<<" , "<<images[i].second<<endl;

        ss<<outdir<<getOSSeparator()<<name<<"."<<type;
        cout<<ss.str()<<endl;

        cv::imshow("preconv",image);
        imgInfo(image);


        cv::Mat image2;

        int maxRange = (1 << 8*(image.elemSize1())) -1;
        cout<<"maxRange: "<<maxRange<<endl;

        image.convertTo(image2, CV_32F, 1.0f/maxRange);
        cv::imshow("afterconv",image2);
        cout<<"converted"<<endl;
        imgInfo(image2);

        cv::waitKey();


        std::vector<int> params;

        if(type == "jpg"){
           params.push_back(CV_IMWRITE_JPEG_QUALITY);
           params.push_back(compr);   // that's percent, so 100 == no compression, 1 == full
        }else{
           params.push_back(CV_IMWRITE_PNG_COMPRESSION);
           params.push_back(compr);   // that's compression level, 9 == full , 0 == none
        }
        cv::imwrite(ss.str(),image,params);

//            cv::Mat image2 = cv::imread(ss.str(),CV_LOAD_IMAGE_COLOR);
//            cv::imshow("compressed",image2);

//            cv::waitKey(2);


    }


}
