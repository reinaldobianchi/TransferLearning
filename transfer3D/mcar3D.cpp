/*
 *  Code adapted from: Ioannis Partalas
 * from http://library.rl-community.org/wiki/Mountain_Car_3D_(CPP)
 *
 *  Based on MontainCar3DSym.cc, created by Matthew Taylor (Based on MountainCar.cc, created by Adam White,
 *  								created on March 29 2007.)
 *
 *  Updated by Reinaldo Bianchi
 *  rbianchi@fei.edu.br
 *  2014
 *  Main Modifications:
 * - Created a class for the Mountain Car 3D
 * - Made it independent of the RL-Glue Library - as much as I like it.
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

#ifndef mcar_INCLUDED
#define mcar_INCLUDED
#include "mcar3D.h"
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



void MCar3D::MCarInit(void)
// Initialize state of Car
{
    mcar_xposition = -0.5;
    mcar_yposition = -0.5;
    
    mcar_xvelocity = 0.0;
    mcar_yvelocity = 0.0;
}

void MCar3D::MCarStep(int a)
// Take action a, update state of car
// this combined the velocity update and position update of Ioannis Partalas original code.
{
    switch(a) {
        case 0:
            mcar_xvelocity += (0)*0.001 + cos(3*mcar_xposition)*(-0.0025);
            mcar_yvelocity += (0)*0.001 + cos(3*mcar_yposition)*(-0.0025);
            break;
        case 1:
            mcar_xvelocity += (-1)*0.001 + cos(3*mcar_xposition)*(-0.0025);
            mcar_yvelocity += (0)*0.001 + cos(3*mcar_yposition)*(-0.0025);
            break;
        case 2:
            mcar_xvelocity += (+1)*0.001 + cos(3*mcar_xposition)*(-0.0025);
            mcar_yvelocity += (0)*0.001 + cos(3*mcar_yposition)*(-0.0025);
            break;
        case 3:
            mcar_xvelocity += (0)*0.001 + cos(3*mcar_xposition)*(-0.0025);
            mcar_yvelocity += (-1)*0.001 + cos(3*mcar_yposition)*(-0.0025);
            break;
        case 4:
            mcar_xvelocity += (0)*0.001 + cos(3*mcar_xposition)*(-0.0025);
            mcar_yvelocity += (+1)*0.001 + cos(3*mcar_yposition)*(-0.0025);
            break;
            
    }
    
    
    if (mcar_xvelocity > mcar_max_velocity) mcar_xvelocity = mcar_max_velocity;
    else if (mcar_xvelocity < -mcar_max_velocity) mcar_xvelocity = -mcar_max_velocity;
    if (mcar_yvelocity > mcar_max_velocity) mcar_yvelocity = mcar_max_velocity;
    else if (mcar_yvelocity < -mcar_max_velocity) mcar_yvelocity = -mcar_max_velocity;
    
    // Below, the position update
    
    mcar_xposition += mcar_xvelocity;
    mcar_yposition += mcar_yvelocity;
    
    if (mcar_xposition > mcar_max_position)
        mcar_xposition = mcar_max_position;
    if (mcar_xposition < mcar_min_position)
        mcar_xposition = mcar_min_position;
    
    if (mcar_yposition > mcar_max_position)
        mcar_yposition = mcar_max_position;
    if (mcar_yposition < mcar_min_position)
        mcar_yposition = mcar_min_position;
    
    if (mcar_xposition==mcar_max_position && mcar_xvelocity>0) mcar_xvelocity = 0;
    if (mcar_xposition==mcar_min_position && mcar_xvelocity<0) mcar_xvelocity = 0;
    
    if (mcar_yposition==mcar_max_position && mcar_yvelocity>0) mcar_yvelocity = 0;
    if (mcar_yposition==mcar_min_position && mcar_yvelocity<0) mcar_yvelocity = 0;
    
}


bool MCar3D::MCarAtGoal ()
// Is Car within goal region?
{
    return (mcar_xposition >= mcar_goal_position && mcar_yposition >= mcar_goal_position);;
}


