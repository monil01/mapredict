// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>


#define Finegrained_RSC_Print

using namespace std;




/*
 * functions declarations end here 
 */

/*
 * Function name: find_mod 
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function finds the mod two double values 
 * 
 * Input: This function takes two arguments 
 * 
 * Output: mod in floating/double 
 * 
 * Description: it just loops through to get the mod 
 */

double 
find_mod(double a, double b) 
{ 
    double mod; 
    // Handling negative values 
    if (a < 0 || b < 0)
    { 
	std::cout << " ERROR: a or b is negative a:" << a << " b:" << b << "\n";
	return 0;
    }
  
    // Finding mod by repeated subtraction 
    while (mod >= b) 
        mod = mod - b; 
  
    return mod; 
} 



