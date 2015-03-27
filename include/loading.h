#ifndef LOADING
#define LOADING

#include <vector>

// for WINDOWS: has to be downloaded and put into ..\Visual Studio 2013\VC\include or similar
// check: http://www.softagalleria.net/download/dirent/
#include <dirent.h>
#include <string>
#include <iostream>


using namespace std;

// parameter processing
template<typename T> bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc)) continue;
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            std::cout<<"PARAM[SET]: "<<param<<" : "<<var<<std::endl;
            return (bool)ss;
        }
    }
    std::cout<<"PARAM[DEF]: "<<param<<" : "<<var<<std::endl;
    return false;
}

static string getOSSeparator() {
#ifdef _WIN32
  return "\\";
#else
  return "/";
#endif
}

static vector<string> getAllImagesFromFolder(const char *dirname) {
  DIR *dir = NULL;
  struct dirent *entry;
  vector<string> allImages;

  dir = opendir(dirname);

  if (!dir) {
    cerr << "Could not open directory " << dirname << ". Exiting..." << endl;
    exit(1);
  }

  const string sep = getOSSeparator();
  string dirStr = string(dirname);

  while(entry = readdir(dir)) {
    if (strstr(entry->d_name, ".png") ||
        strstr(entry->d_name, ".jpg") ||
        strstr(entry->d_name, ".tif")) {
      string fileName(entry->d_name);
      string fullPath = dirStr + sep + fileName;
      allImages.push_back(fullPath);
    }
  }
  closedir(dir);

  // sort string alphabetically
  std::sort(allImages.begin(), allImages.end());
  return allImages;
}

static vector< pair<string,string> > getAllImagesFromFolder2(const char *dirname) {
  DIR *dir = NULL;
  struct dirent *entry;
  vector< pair<string,string> > allImages;

  dir = opendir(dirname);

  if (!dir) {
    cerr << "Could not open directory " << dirname << ". Exiting..." << endl;
    exit(1);
  }

  const string sep = getOSSeparator();
  string dirStr = string(dirname);

  while(entry = readdir(dir)) {
    if (strstr(entry->d_name, ".png") ||
        strstr(entry->d_name, ".jpg") ||
        strstr(entry->d_name, ".tif")) {
      string fileName(entry->d_name);
      string fullPath = dirStr + sep + fileName;
      allImages.push_back(std::make_pair(fullPath,fileName));
    }
  }
  closedir(dir);

  // sort string alphabetically
  std::sort(allImages.begin(), allImages.end());
  return allImages;
}

#endif // LOADING

