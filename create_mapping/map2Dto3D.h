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
#include <random>

#ifndef DEFS_INCLUDED
#define DEFS_INCLUDED
#include "definitions.h"
#endif

#ifndef MCAR2D_INCLUDED
#define MCAR2D_INCLUDED
#include "mcar2D.h"
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



double time_init;



double nnWeights[5][3];             // This declares the neural network weight matrix

double delta;                       // For the NN update

double bias = -0.01;                // Bias for the NN.

double learningrate = 0.9;          // learning rate

int MAX_ITERACTIONS = 500000;       // If you increase this value, it will take more time,
                                    // but you will be able to see more clearly the learning.
                                    // 10 Million is a good number to see the larger values.
                                    // But has no effect on mapping results - remember to increase the bias by the same amount.
                                    // if you decrease this value, it may not learn the mapping.

double sum;                         // For the sum of the neuron output.
