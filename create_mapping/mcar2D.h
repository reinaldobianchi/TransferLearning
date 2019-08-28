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

class MCar2D
{
public:
    // Atributes
    float mcar_position, mcar_velocity; // Car position and speed
    // Methods

    void MCarInit(void);
    void MCarStep(int a);
    bool MCarAtGoal (void);
};

