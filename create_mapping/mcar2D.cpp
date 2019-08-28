/*
 This is an example program for reinforcement learning with linear
 function approximation.  The code follows the psuedo-code for linear,
 gradient-descent Sarsa(lambda) given in Figure 8.8 of the book
 "Reinforcement Learning: An Introduction", by Sutton and Barto.
 One difference is that we use the implementation trick mentioned on
 page 189 to only keep track of the traces that are larger
 than "min-trace".
 
 Original code can be obtained here:
 http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar.html
 
 Written by Rich Sutton 12/19/00
 
 
 Code updated, modified and "improved" by Reinaldo A. C. Bianchi
 Department of Electrical Engieering
 Centro Universitario da FEI
 Sao Bernardo do Campo - Brazil
 rbianchi@fei.edu.br
 http://fei/edu/br/~rbianchi
 Last update: September 2014
 
 Main Modifications:
 - Created the Mcar 2D Class, where the mcar related methods asre.
 */

/********************************************************************
 *                                                                  *
 *                        LIBRARY INCLUDES                          *
 *                                                                  *
 ********************************************************************/

#ifndef DEFS_INCLUDED
#define DEFS_INCLUDED
#include "definitions.h"
#endif

#ifndef MCAR2D_INCLUDED
#define MCAR2D_INCLUDED
#include "mcar2D.h"
#endif



/********************************************************************
 *                                                                  *
 *              DEFINITION OF VARIABLES AND PARAMETERS              *
 *                                                                  *
 ********************************************************************/




/********************************************************************
 *                                                                  *
 *                CODE IMPLEMENTATION START HERE                    *
 *                                                                  *
 ********************************************************************/



//////////     Part 1: Mountain Car code     //////////////



void MCar2D::MCarInit(void)
// Initialize state of Car
{
    mcar_position = -0.5;
    mcar_velocity = 0.0;
}

void MCar2D::MCarStep(int a)
// Take action a, update state of car
{
    mcar_velocity += (a-1)*0.001 + cos(3*mcar_position)*(-0.0025);
    if (mcar_velocity > mcar_max_velocity) mcar_velocity = mcar_max_velocity;
    if (mcar_velocity < -mcar_max_velocity) mcar_velocity = -mcar_max_velocity;
    mcar_position += mcar_velocity;
    if (mcar_position > mcar_max_position) mcar_position = mcar_max_position;
    if (mcar_position < mcar_min_position) mcar_position = mcar_min_position;
    if (mcar_position==mcar_min_position && mcar_velocity<0) mcar_velocity = 0;
}

bool MCar2D::MCarAtGoal ()
// Is Car within goal region?
{
    return mcar_position >= mcar_goal_position;
}


