// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ### Final Project: Variational Depth from Focus
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2014, September 8 - October 10
// ###
// ###
// ### Maria Klodt, Jan Stuehmer, Mohamed Souiai, Thomas Moellenhoff
// ###
// ###

// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070

#ifndef _WINTIME_H_
#define	_WINTIME_H_

/*
 * Redefinition of clock_gettime (nonstandard timer call) for windows
 * http://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows/5404467#5404467
 * with minor modifications from timeval to timespec and converting micro- into nanoseconds
 */

#include <Windows.h>
//#include <time.h>

namespace vdff {
  /*
   * Structure defined by POSIX.1b to be like a timeval.
   */
  struct timespec {
    time_t	tv_sec;		/* seconds */
    long	tv_nsec;	/* and nanoseconds */
  };


  LARGE_INTEGER getFILETIMEoffset();

  int clock_gettime(struct timespec *tv);
}

#endif 
