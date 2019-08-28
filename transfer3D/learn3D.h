/*
 
 This program creates the mapping between the 2D and the 3D mountain car problem.
 
 By Reinaldo Bianchi
 rbianchi@fei.edu.br
 2014
 
 */

/********************************************************************
 *                                                                  *
 *                        LIBRARY INCLUDES                          *
 *                                                                  *
 ********************************************************************/

#include <iostream>
#include <fstream>
#include <cmath>
//#include <random>

#ifndef DEFS_INCLUDED
#define DEFS_INCLUDED
#include "definitions.h"
#endif


#ifndef MCAR3D_INCLUDED
#define MCAR3D_INCLUDED
#include "mcar3D.h"
#endif


using namespace std;




/********************************************************************
 *                                                                  *
 *              DEFINITION OF VARIABLES AND PARAMETERS              *
 *                                                                  *
 ********************************************************************/


#define NUM_RUNS 30
#define NUM_EPISODES 1100
#define MAX_STEPS 5000

MCar3D mcar;

