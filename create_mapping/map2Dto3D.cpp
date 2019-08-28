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



#ifndef MAPPING_INCLUDED
#define MAPPING_INCLUDED
#include "map2Dto3D.h"
#endif



/********************************************************************
 *                                                                  *
 *                CODE IMPLEMENTATION START HERE                    *
 *                                                                  *
 ********************************************************************/




//////////////// Timer functions /////////

/* Restarts o timer; */
void init_timer()
{
    time_init = (float)clock()/(float) CLOCKS_PER_SEC;
}

/* record the time it took for each run. */
void end_timer()
{
    printf("This run number took %f seconds\n",  ((float)clock()/(float) CLOCKS_PER_SEC) - time_init);
}


/********************************************************************
 *                                                                  *
 *                           MAIN LOOP                              *
 *                                                                  *
 ********************************************************************/



int main(void)
{

    // create the mountain car objetcts
    MCar2D mc2d;
    MCar3D mc3d;
    
    
    int i,j,k;
    int action2d, action3d;
    
    double xi=0, yi=0;
    
    double old_mcar_velocity=0, old_mcar3d_xvelocity=0, old_mcar3d_yvelocity=0;
    
    double variation_2d =0, variation_3dx=0, variation_3dy=0;

    
    // this initialize the weights as zero.
    for (i =0; i<5;i++)
        for (j =0; j<3; j++)
            nnWeights[i][j] = 0;
    
    
    
    // Define and Initialize the Random Generators.the
    // Create source of randomness, and initialize it with non-deterministic seed
    // Will use Mersenee twister engine 64 bits http://www.cplusplus.com/reference/random/mersenne_twister_engine/
    std::mt19937_64 eng((std::random_device())());
    
    // A distribution that takes randomness and produces values in specified range
    
    // One for action 2d
    std::uniform_int_distribution<> dist_action2d(0, 2);
    
    // One for action 3d
    std::uniform_int_distribution<> dist_action3d(0, 4);
    
    // Init both mcars
    mc2d.MCarInit();
    mc3d.MCarInit();
    
    std::cout.precision(5);
    
    
    init_timer();
    
    // Start learning
    
    for (i = 0; i < MAX_ITERACTIONS; i++)
        
    {
        // Randomly select action
        action2d = dist_action2d(eng);
        action3d = dist_action3d(eng);
        
        // Se effects on both Mcars
        
        mc2d.MCarStep(action2d);
        mc3d.MCarStep(action3d);
        
        
        // Neural network training here
        
        // First, we have to compute the variations that happaned in both source and target domains.
        
        // Compute the Variation in the source domain
        variation_2d = mc2d.mcar_velocity - old_mcar_velocity;
        
        // Compute the Variation in the target domain
        variation_3dx = mc3d.mcar_xvelocity - old_mcar3d_xvelocity;
        variation_3dy = mc3d.mcar_yvelocity - old_mcar3d_yvelocity;
        
        
        // we could use directly the action taken to find out what variation use, but this would mean to code in the actions...
        // to make the mapping more generic, i.e., not depend on knowledge of the actions, we will compare the results.
        
        // now, we know that, in the target domain, only one of the 2 variations can happen, because only one action was taken.
        // as we have 2 dimensions, only one of then can change with one action executed.
        // so we must use the value that has some variation.
        // if the action used is no action, this value will be zero.
        
        if (abs(variation_3dx) > abs(variation_3dy))
            xi=variation_3dx;
        else xi = variation_3dy;
        
        // the variation in the source domain depends only of one dimention, so it must be...
        yi = variation_2d;
        
        
        
        
        // this computed the delta, using hebbian learning
        // unsupervised learning: reinforce pairs that have the same value, punish pairs with different values.
        delta = learningrate * xi * yi;
        
        // this updates the weight that is connectiong the 2 actions that were performed
        nnWeights[action3d][action2d] += delta;
        
        
        
        // Print effects
      //  printf("%d %d %f \t %f \t %f \t %f\n", action2d, action3d, variation_2d, variation_3dx, variation_3dy, delta);
        
        
        // Updates the old vars to be able to compute the variations in the next time step
        old_mcar_velocity = mc2d.mcar_velocity;
        old_mcar3d_xvelocity = mc3d.mcar_xvelocity;
        old_mcar3d_yvelocity = mc3d.mcar_yvelocity;
        
        // if any of the domains reach its final position, we must restart it.
        
        if (mc2d.mcar_position >= mcar_goal_position) mc2d.MCarInit();
        if (mc3d.mcar_xposition >= mcar_goal_position) mc3d.MCarInit();
        if (mc3d.mcar_yposition >= mcar_goal_position) mc3d.MCarInit();
        
    }
    
    end_timer();
    
    
    // Now we print the neural connections made.

    cout << "\nThis is the weights that the NN learned.\n";
    for (i =0; i<5;i++)
    {
        for (j =0; j<3; j++)
        {
            cout << nnWeights[i][j] << "\t";
        }
        cout << "\n";
    }
    
    // now we print the table of mapping between the actions.
    // to do this, we present each target domain action to the neural network, and compute the results.
    
    
    // entradas
    int nnInputs [5][5] = {{ 1, -1, -1, -1, -1},{ -1, 1, -1, -1, -1},{ -1, -1, 1, -1, -1},{ -1, -1, -1, 1, -1},{ -1, -1, -1, -1, 1}};
    
    
    
   
    cout << "\nThis is the mapping between actions, and the weighted sum.\n";
    // for all the 5 input actions
    for (i =0; i<5;i++)
    {
        //compute output 1 to 3
        for (j =0; j<3; j++)
        {
            sum = bias;
            for (k  =0; k <5;k ++)
            {
                sum += nnWeights[k][j] * nnInputs[i][k];
                
            }
            // this is the neuron activation function
            if (sum > 0.0) cout << "+1 ";
            else cout << "-1 ";
            cout <<  std::fixed;
            cout << "(" << sum << ") \t";
            
        }
        cout << "\n";
        
    }
    
   // This saves the mapping into a file.
    ofstream resultfile;
    resultfile.open ("mapping.txt");
    
    // for all the 5 input actions
    for (i =0; i<5;i++)
    {
        //compute output 1 to 3
        for (j =0; j<3; j++)
        {
            sum = bias;
            for (k  =0; k <5;k ++)
            {
                sum += nnWeights[k][j] * nnInputs[i][k];
                
            }
            // this is the neuron activation function
            if (sum > 0.0) resultfile << "+1 ";
            else resultfile << "-1 ";
            
        }
        resultfile << "\n";
        
    }
    resultfile.close();
    
    
    
    
    // this ENDS the Program. DO NOT REMOVE THIS RETURN
    return 0;
}

