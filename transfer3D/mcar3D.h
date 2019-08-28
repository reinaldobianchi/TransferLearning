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

#include <math.h>

/********************************************************************
 *                                                                  *
 *              DEFINITION OF VARIABLES AND PARAMETERS              *
 *                                                                  *
 ********************************************************************/





/********************************************************************
 *                                                                  *
 *                  CLASS DEFINITION START HERE                     *
 *                                                                  *
 ********************************************************************/

class MCar3D
{
public:
    // Atributes
    float mcar_xposition, mcar_yposition, mcar_xvelocity, mcar_yvelocity; // Car position and speed
    // Methods

    void MCarInit(void);
    void MCarStep(int a);
    bool MCarAtGoal (void);
};

