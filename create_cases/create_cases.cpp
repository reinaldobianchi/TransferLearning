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
 - Joined all files in one, to make compilation easier - there
 is no need to download other files
 - added Q-Learning algorithm.
 - Re-structured the file, to have all definitions and variable
 values at the top of the file
 - Added timer
 - Added code to generate cases from the learned Q-function approximator.
 - Changed output file for easy importing into Matlab, for Graph Generation
 - added decai in epsilon greedy rule.
 
 */

/********************************************************************
 *                                                                  *
 *                        LIBRARY INCLUDES                          *
 *                                                                  *
 ********************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <random>

#ifndef O_BINARY
#define O_BINARY 0
#endif

using namespace std;


/********************************************************************
 *                                                                  *
 *              DEFINITION OF VARIABLES AND PARAMETERS              *
 *                                                                  *
 ********************************************************************/




///////////            For the Tiles Code            /////////////////

#define MAX_NUM_VARS 20        // Maximum number of variables in a grid-tiling


//////////     Part 1: Mountain Car code     //////////////

// Global variables:
float mcar_position, mcar_velocity;

#define mcar_min_position -1.2
#define mcar_max_position 0.6
#define mcar_max_velocity 0.07                      // the negative of this in the minimum velocity
#define mcar_goal_position 0.5

#define POS_WIDTH (1.7 / 8)                         // the tile width for position
#define VEL_WIDTH (0.14 / 8)                        // the tile width for velocity

#define NUM_RUNS 30
#define NUM_EPISODES 200

#define MAX_STEPS 10000

//////////     Part 2: Semi-General RL code     //////////////


#define SARSA 1                    					// If 0, Q-Learning is used. If 1, SARSA is used

#define MEMORY_SIZE 10000                           // number of parameters to theta, memory size
#define NUM_ACTIONS 3                               // number of actions
#define NUM_TILINGS 10

// Global RL variables:
float Q[NUM_ACTIONS];                               // action values
float theta[MEMORY_SIZE];                           // modifyable parameter vector, aka memory, weights
float e[MEMORY_SIZE];                               // eligibility traces
int F[NUM_ACTIONS][NUM_TILINGS];                    // sets of features, one set per action

// Standard RL parameters:
#define epsilon 0.0                                 // probability of random action
#define epsilon_decay 0.0                           // decay in the epsilon: in this way, the last episode is made in a completely greedy way.  epsilon/(NUM_EPISODES-1)
#define alpha 0.5                                   // step size parameter
#define lambda 0.9                                  // trace-decay parameters
#define gamma 1.0                                   // discount-rate parameters

// Global stuff for efficient eligibility trace implementation:
#define MAX_NONZERO_TRACES 1000
int nonzero_traces[MAX_NONZERO_TRACES];
int num_nonzero_traces = 0;
int nonzero_traces_inverse[MEMORY_SIZE];
float minimum_trace = 0.01;

// Variables for file manipulation and recording the results.


FILE *record_file;
FILE * record_file_cases;
char filename[] = "results.txt";                    // This is the name of the output file

#define MATLAB_OUT 1                                // If zero, output is as Professor Sutton originally wrote.
                                                    // If 1, it outputs a matrix to be read by matlab/octave function 'load'.

// Variables for timing the execution of the larning code.

double tempo[NUM_RUNS];
double time_init;




/********************************************************************
 *                                                                  *
 *               PROTOTYPES FOR ALL FUNCTIONS USED                  *
 *                                                                  *
 ********************************************************************/



//////////     For the Tiles Code     //////////////

void GetTiles(int tiles[],int num_tilings,float variables[], int num_variables,
              int memory_size,int hash1=-1, int hash2=-1, int hash3=-1);

int hash_coordinates(int *coordinates, int num_indices, int memory_size);


//////////     Part 1: Mountain Car code     //////////////

// Profiles
void MCarInit();                              // initialize car state
void MCarStep(int a);                         // update car state for given action
bool MCarAtGoal ();                           // is car at goal?


//////////     Part 2: Semi-General RL code     //////////////

// Profiles:
int episode(int max_steps, int episode_num);        // do one episode, return length
void LoadQ();                                       // compute action values for current theta, F
void LoadQ(int a);                                  // compute one action value for current theta, F
void LoadF();                                       // compute feature sets for current state
int argmax(float Q[NUM_ACTIONS]);                   // compute argmax action from Q
bool WithProbability(float p);                      // helper - true with given probability
void ClearTrace(int f);                             // clear or zero-out trace, if any, for given feature
void ClearExistentTrace(int f, int loc);            // clear trace at given location in list of nonzero-traces
void DecayTraces(float decay_rate);                 // decay all nonzero traces
void SetTrace(int f, float new_trace_value);        // set trace to given value
void IncreaseMinTrace();                            // increase minimal trace value, forcing more to 0, making room for new ones
void WriteTheta(char *filename);                    // write weights (theta) to a file
void ReadTheta(char *filename);                     // reads them back in




//////////////// For Timer functions /////////

void init_timer();
void end_timer(int iteracao);
void save_time_results();


///////////////// For File Manipulation   ///////////
void open_record_file(char *filename);
void start_record();
void end_record();


/********************************************************************
 *                                                                  *
 *                CODE IMPLEMENTATION START HERE                    *
 *                                                                  *
 ********************************************************************/



//////////     Part 1: Mountain Car code     //////////////


void MCarInit()
// Initialize state of Car
{
    mcar_position = -0.5;
    mcar_velocity = 0.0;
}

void MCarStep(int a)
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

bool MCarAtGoal ()
// Is Car within goal region?
{
    return mcar_position >= mcar_goal_position;
}





//////////     Part 2: Semi-General RL code     //////////////


int episode(int max_steps, int episode_num)
// Runs one episode of at most max_steps, returning episode length; see Figure 8.8 of RLAI book
{
    MCarInit();                                                  // initialize car's state
    DecayTraces(0.0);                                            // clear all traces
    LoadF();                                                     // compute features
    LoadQ();                                                     // compute action values
    
    // This is the First Acrion Selection
    int action = argmax(Q);                                      // pick argmax action
    if (WithProbability((epsilon-epsilon_decay*episode_num))) action = rand() % NUM_ACTIONS; // ...or maybe pick action at random
    
    int step = 0;
    do {
        step++;                                                  // now do a bunch of steps
        DecayTraces(gamma*lambda);                               // let traces fall
        for (int a=0; a<NUM_ACTIONS; a++)                        // optionally clear other traces
            if (a != action)
            {
                for (int j=0; j<NUM_TILINGS; j++)
                    ClearTrace(F[a][j]);
            }
        for (int j=0; j<NUM_TILINGS; j++) SetTrace(F[action][j],1.0); // replace traces
        MCarStep(action);                                        // actually take action
        float reward = -1;
        float delta = reward - Q[action];
        LoadF();                                                 // compute features new state
        LoadQ();                                                 // compute new state values
        action = argmax(Q);
        
        // This line defines if SARSA or Q-Learning is used.
        // If SARSA, the action used to update is on-policy,
        // that is, is the actual action chosen, including random moves.
        // If SARSA == 0, this line is ignored, and update is done with argmax(Q)
        // thus implementing the Q-Learning Algorithm update
        
        if (SARSA)
            if (WithProbability(epsilon-epsilon_decay*episode_num)) action = rand() % NUM_ACTIONS;
        
        if (!MCarAtGoal()) delta += gamma * Q[action];
        float temp = (alpha/NUM_TILINGS)*delta;
        for (int i=0; i<num_nonzero_traces; i++)                 // update theta (learn)
        {
            int f = nonzero_traces[i];
            theta[f] += temp * e[f];
        }
        LoadQ(action);
		  }
    while (!MCarAtGoal() && step<max_steps);                     // repeat until goal or time limit
    return step;
}                                                // return episode length

void LoadQ()
// Compute all the action values from current F and theta
{
    for (int a=0; a<NUM_ACTIONS; a++)
    {
        Q[a] = 0;
        for (int j=0; j<NUM_TILINGS; j++) Q[a] += theta[F[a][j]];
		  }
}

void LoadQ(int a)
// Compute an action value from current F and theta
{
    Q[a] = 0;
    for (int j=0; j<NUM_TILINGS; j++) Q[a] += theta[F[a][j]];
}

void LoadF()
// Compute feature sets for current car state
{
    float state_vars[2];
    state_vars[0] = mcar_position / POS_WIDTH;
    state_vars[1] = mcar_velocity / VEL_WIDTH;
    for (int a=0; a<NUM_ACTIONS; a++)
        GetTiles(&F[a][0],NUM_TILINGS,state_vars,2,MEMORY_SIZE,a);
}

int argmax(float Q[NUM_ACTIONS])
// Returns index (action) of largest entry in Q array, breaking ties randomly
{
    int best_action = 0;
    float best_value = Q[0];
    int num_ties = 1;                               // actually the number of ties plus 1
    for (int a=1; a<NUM_ACTIONS; a++)
    {
        float value = Q[a];
        if (value >= best_value)
        {
            if (value > best_value)
            {
                best_value = value;
                best_action = a;
            }
            else
            {
                num_ties++;
                if (0 == rand() % num_ties)
                { best_value = value;
                    best_action = a;
                }
            }
        }
    };
    return best_action;
}

int argmin(float Q[NUM_ACTIONS])
// Returns index (action) of lowest entry in Q array, breaking ties randomly
{
    int worst_action = 0;
    float worst_value = Q[0];
    int num_ties = 1;                               // actually the number of ties plus 1
    for (int a=1; a<NUM_ACTIONS; a++)
    {
        float value = Q[a];
        if (value <= worst_value)
        {
            if (value < worst_value)
            {
                worst_value = value;
                worst_action = a;
            }
            else
            {
                num_ties++;
                if (0 == rand() % num_ties)
                { worst_value = value;
                    worst_action = a;
                }
            }
        }
    };
    return worst_action;
}



bool WithProbability(float p)
// Returns TRUE with probability p
{
    return p > ((float)rand()) / RAND_MAX;
}

// Traces code:

/* Below is the code for selectively working only with traces >= minimum_trace.
 Other traces are forced to zero.  We keep a list of which traces are nonzero
 so that we can work only with them.  This list is implemented as the array
 "nonzero_traces" together with its length "num_nonzero_traces".  When a trace
 falls below minimum_trace and is forced to zero, we remove it from the list by
 decrementing num_nonzero_traces and moving the last element into the "hole"
 in nonzero_traces made by this one that we are removing.  A final complication
 arises because sometimes we want to clear (set to zero and remove) a trace
 but we don't know its position within the list of nonzero_traces.  To avoid
 havint to search through the list we keep inverse pointers from each trace
 back to its position (if nonzero) in the nonzero_traces list.  These inverse
 pointers are in the array "nonzero_traces_inverse".
 
 Global stuff (really at top of file) for efficient eligibility trace implementation:
 #define MAX_NONZERO_TRACES 1000
 int nonzero_traces[MAX_NONZERO_TRACES];
 int num_nonzero_traces = 0;
 int nonzero_traces_inverse[MEMORY_SIZE];
 float minimum_trace = 0.01;  */

void ClearTrace(int f)
// Clear any trace for feature f
{
    if (!(e[f]==0.0))
        ClearExistentTrace(f,nonzero_traces_inverse[f]);
}

void ClearExistentTrace(int f, int loc)
// Clear the trace for feature f at location loc in the list of nonzero traces
{
    e[f] = 0.0;
    num_nonzero_traces--;
    nonzero_traces[loc] = nonzero_traces[num_nonzero_traces];
    nonzero_traces_inverse[nonzero_traces[loc]] = loc;
}

void DecayTraces(float decay_rate)
// Decays all the (nonzero) traces by decay_rate, removing those below minimum_trace
{
    for (int loc=num_nonzero_traces-1; loc>=0; loc--)      // necessary to loop downwards
    {
        int f = nonzero_traces[loc];
        e[f] *= decay_rate;
        if (e[f] < minimum_trace) ClearExistentTrace(f,loc);
		  }
}

void SetTrace(int f, float new_trace_value)
// Set the trace for feature f to the given value, which must be positive
{
    if (e[f] >= minimum_trace) e[f] = new_trace_value;         // trace already exists
    else {
        while (num_nonzero_traces >= MAX_NONZERO_TRACES) IncreaseMinTrace(); // ensure room for new trace
        e[f] = new_trace_value;
        nonzero_traces[num_nonzero_traces] = f;
        nonzero_traces_inverse[f] = num_nonzero_traces;
        num_nonzero_traces++;
		  }
}

void IncreaseMinTrace()
// Try to make room for more traces by incrementing minimum_trace by 10%,
// culling any traces that fall below the new minimum
{
    minimum_trace += 0.1 * minimum_trace;
    cout << "Changing minimum_trace to " << minimum_trace << endl;
    for (int loc=num_nonzero_traces-1; loc>=0; loc--)      // necessary to loop downwards
    {
        int f = nonzero_traces[loc];
        if (e[f] < minimum_trace) ClearExistentTrace(f,loc);
		  };
}

void WriteTheta(char *filename)
// writes parameter vector theta to a file as binary data; writes over any old file
{
    int file = open(filename, O_BINARY | O_CREAT | O_WRONLY);
    write(file, (char *) theta, MEMORY_SIZE * sizeof(float));
    close(file);
}

void ReadTheta(char *filename)
// reads parameter vector theta from a file as binary data
{
    int file = open(filename, O_BINARY | O_RDONLY);
    read(file, (char *) theta, MEMORY_SIZE * sizeof(float));
    close(file);
}





// PART 4 TILES CODE FROM TILES.C

/*
 
 Below is R. Sutton comments ont this part of the code. it is from 2000, and is outdated.
 
 External documentation and recommendations on the use of this code is
 available at http://www.cs.umass.edu/~rich/tiles.html.
 
 This is an implementation of grid-style tile codings, based originally on
 the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm).
 Here we provide a procedure, "GetTiles", that maps real
 variables to a list of tiles. This function is memoryless and requires no
 setup. We assume that hashing colisions are to be ignored. There may be
 duplicates in the list of tiles, but this is unlikely if memory-size is
 large.
 
 The input variables will be gridded at unit intervals, so generalization
 will be by 1 in each direction, and any scaling will have
 to be done externally before calling tiles.
 
 It is recommended by the UNH folks that num-tiles be a power of 2, e.g., 16.
 */



void GetTiles(
              int tiles[],               // provided array contains returned tiles (tile indices)
              int num_tilings,           // number of tile indices to be returned in tiles
              float variables[],         // array of variables
              int num_variables,         // number of variables
              int memory_size,           // total number of possible tiles (memory size)
              int hash1,                 // change these from -1 to get a different hashing
              int hash2,
              int hash3)
{
    int i,j;
    int qstate[MAX_NUM_VARS];
    int base[MAX_NUM_VARS];
    int coordinates[MAX_NUM_VARS + 4];   /* one interval number per rel dimension */
    int num_coordinates;
    
    if (hash1 == -1)
        num_coordinates = num_variables + 1;       // no additional hashing corrdinates
    else if (hash2 == -1) {
        num_coordinates = num_variables + 2;       // one additional hashing coordinates
        coordinates[num_variables+1] = hash1;
    }
    else if (hash3 == -1) {
        num_coordinates = num_variables + 3;       // two additional hashing coordinates
        coordinates[num_variables+1] = hash1;
        coordinates[num_variables+2] = hash2;
    }
    else {
        num_coordinates = num_variables + 4;       // three additional hashing coordinates
        coordinates[num_variables+1] = hash1;
        coordinates[num_variables+2] = hash2;
        coordinates[num_variables+3] = hash3;
    }
    
    /* quantize state to integers (henceforth, tile widths == num_tilings) */
    for (i = 0; i < num_variables; i++) {
        qstate[i] = (int) floor(variables[i] * num_tilings);
        base[i] = 0;
    }
    
    /*compute the tile numbers */
    for (j = 0; j < num_tilings; j++) {
        
        /* loop over each relevant dimension */
        for (i = 0; i < num_variables; i++) {
            
            /* find coordinates of activated tile in tiling space */
            if (qstate[i] >= base[i])
                coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings);
            else
                coordinates[i] = qstate[i]+1 + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings;
				        
            /* compute displacement of next tiling in quantized space */
            base[i] += 1 + (2 * i);
        }
        /* add additional indices for tiling and hashing_set so they hash differently */
        coordinates[i++] = j;
        
        tiles[j] = hash_coordinates(coordinates, num_coordinates, memory_size);
    }
    return;
}


/* hash_coordinates
 Takes an array of integer coordinates and returns the corresponding tile after hashing
 */
int hash_coordinates(int *coordinates, int num_indices, int memory_size)
{
    static int first_call = 1;
    static unsigned int rndseq[2048];
    int i,k;
    long index;
    long sum = 0;
    
    /* if first call to hashing, initialize table of random numbers */
    if (first_call) {
        for (k = 0; k < 2048; k++) {
            rndseq[k] = 0;
            for (i=0; i < sizeof(int); ++i)
                rndseq[k] = (rndseq[k] << 8) | (rand() & 0xff);
        }
        first_call = 0;
    }
    
    for (i = 0; i < num_indices; i++) {
        /* add random table offset for this dimension and wrap around */
        index = coordinates[i];
        index += (449 * i);
        index %= 2048;
        while (index < 0) index += 2048;
        
        /* add selected random number to sum */
        sum += (long)rndseq[(int)index];
    }
    index = (int)(sum % memory_size);
    while (index < 0) index += memory_size;
    
    return(index);
}




//////////////// Timer functions /////////

/* Restarts o timer; */
void init_timer()
{
    time_init = (float)clock()/(float) CLOCKS_PER_SEC;
}

/* record the time it took for each run. */
void end_timer(int iteracao)
{
    
    tempo[iteracao]  = ((float)clock()/(float) CLOCKS_PER_SEC) - time_init;
    
    printf("This run number %d took %f seconds\n",  iteracao, tempo[iteracao]);
}

// Computes meand and std deviation for the time
void save_time_results()
{
    FILE *run;
    int j, r;
    char fname[20];
    
    int t;
    double sum, media, desvio_quad;
    
    
    
    // Prints the average
    
    run = fopen("mean-run-time.txt", "w");
    
    sum = 0;
    for (r=0; r<NUM_RUNS; r++)
        sum += tempo[r];
    
    media = sum/(double)NUM_RUNS;
    // Calcula o desvio PG 16 DO OTAVIANO HELENE,eh o sigma ^ 2)
    
    desvio_quad = 0;
    for (r = 0; r < NUM_RUNS; r++)
        desvio_quad += pow((tempo[r] - media), 2);
    
    desvio_quad = desvio_quad / (double) (NUM_RUNS);
    
    // Imprime media e desvio
    printf("\nMean Time %f std deviation %f \n\n",  media , sqrt(desvio_quad)) ;
    fprintf(run,"Mean Time %f std deviation %f \n",  media , sqrt(desvio_quad)) ;
    
    fclose(run);
}



///////////   File manipulation and writting stuff ///////////////////




void open_record_file(char *filename)
{
    
    if (MATLAB_OUT)
    {
        if ( (record_file=fopen(filename,"w")) == NULL)
        {
            printf("Could not open file for output\nBailing out!\n");
            exit (EXIT_FAILURE);
        }
    }
    else
    {
        if ( (record_file=fopen(filename,"r")) == NULL)
        {
            fclose(record_file);
            record_file=fopen(filename,"a");
            fprintf(record_file,"(:record-fields :MEMORY_SIZE :NUM_ACTIONS :runs :episodes :alg :updating :alpha :lambda :epsilon :gamma :data)\n");}
        else
        {
            fclose(record_file);
            record_file=fopen(filename,"a");
        }
        
    }
}



void start_record()
{
    
    if (!MATLAB_OUT)
        fprintf(record_file, "(%d %d %d %d :SARSA :ONLINE %f %f %f %f (",
                MEMORY_SIZE, NUM_ACTIONS, NUM_RUNS, NUM_EPISODES, alpha, lambda, epsilon, gamma);
}

void end_record()
{
    if (!MATLAB_OUT)
        fprintf(record_file,"))\n");
    fclose (record_file);
}




/********************************************************************
 *                                                                  *
 *                           MAIN LOOP                              *
 *                                                                  *
 ********************************************************************/

//////////     Part 3: Top-Level code     //////////////
// (outer loops, data collection, print statements, frequently changed stuff)



int main()
// The main program just does a bunch of runs, each consisting of some episodes.
{
    
    //////////////////////////////////////////////////////////////////////
    //    Here we do the learning.
    //////////////////////////////////////////////////////////////////////
    
    
    /* initialize random seed:
     Added By Reinaldo in 2014 */
    srand (time(NULL));
    
    
    open_record_file(filename);
    for (int i=0; i<MEMORY_SIZE; i++)
        e[i]= 0.0;         // traces must start at 0
    start_record();
    for (int run=0; run<NUM_RUNS; run++)
    {
        
        
        cout << "Starting run #" << run << endl;
        
        
        init_timer();
        
        for (int i=0; i<MEMORY_SIZE; i++)
            theta[i]= 0.0; // clear memory at start of each run
        if (!MATLAB_OUT)  fprintf(record_file," (");
        for (int episode_num=0; episode_num<NUM_EPISODES; episode_num++)
            fprintf(record_file, "%d ",episode(MAX_STEPS, episode_num));
        
        if (MATLAB_OUT) fprintf(record_file, " \n");
        else fprintf(record_file, ") ");
        
        end_timer(run);
        
        
    }
    end_record();
    save_time_results();
    
    //////////////////////////////////////////////////////////////////////
    //    Here we start to collect the cases to be transfered.
    //    If you are only interested in que mountain Car problem, you can
    //    cut away all the code that is below.
    //////////////////////////////////////////////////////////////////////
    
    // Define the number of cases to be collected
    int NCASES = 500;
    
    int action;
    
    
    // Define and Initialize the Random Generators.the
    // Create source of randomness, and initialize it with non-deterministic seed
    // Will use Mersenee twister engine 64 bits http://www.cplusplus.com/reference/random/mersenne_twister_engine/
    std::mt19937_64 eng((std::random_device())());
    
    // A distribution that takes randomness and produces values in specified range
    
    
    // One for position. This is the range to sample: from -1.2 to 0.6.
    std::uniform_real_distribution<> dist_pos(-1.2, 0.6);
    
    // One for velocity
    std::uniform_real_distribution<> dist_vel(-0.07, 0.07);

    // One for action
    std::uniform_int_distribution<> dist_action(0, 2);
    
    
    //////////////////////////////////////////////////////////////////////
    // First method: select NCASES cases from random sampling the state and finding the best action for that state.
    //////////////////////////////////////////////////////////////////////
    
    if ( (record_file_cases=fopen("cases-argmax.txt","w")) == NULL)
    {
        printf("Could not open file for output\nBailing out!\n");
        exit (EXIT_FAILURE);
    }
    
    
    init_timer();
    for (int n = 0; n < NCASES; n++)
    {
        // Randomly select state
        mcar_position = dist_pos(eng);
        mcar_velocity = dist_vel(eng);
        
        LoadF();        // Compute feature sets for current car state
        LoadQ();        // Compute all the action values from current F and theta
        
        // Selects the best action
        action = argmax(Q);

        // save S,A,Q in file.
        fprintf(record_file_cases, "%f %f %d %f\n", mcar_position, mcar_velocity, action, Q[action]);
        
        
    }
    fclose (record_file_cases);
    
    end_timer(-1);
    
    
    
    //////////////////////////////////////////////////////////////////////
    // Second method: select NCASES cases from random sampling the state and
    // action, discarding the case if the action corresponds to the worst action.
    //////////////////////////////////////////////////////////////////////
    
    if ( (record_file_cases=fopen("cases-luizao.txt","w")) == NULL)
    {
        printf("Could not open file for output\nBailing out!\n");
        exit (EXIT_FAILURE);
    }
    
      init_timer();
    for (int n = 0; n < NCASES; n++)
    {
        // Randomly select state
        mcar_position = dist_pos(eng);
        mcar_velocity = dist_vel(eng);
        // Randomly select action
        action = dist_action(eng);
        
        
        LoadF();        // Compute feature sets for current car state
        LoadQ();        // Compute all the action values from current F and theta
        
        // verifies if the action chosen is not the one that has the worst Q Value
        if (action != argmin(Q))
        {
            // save S,A,Q in file.
            fprintf(record_file_cases, "%f %f %d %f \n", mcar_position, mcar_velocity, action, Q[action]);
        }
        
        else {
            n--;
            //  printf( " For Debug - Bad case: %f %f %d %f \n", mcar_position, mcar_velocity, action, Q[action]);
        }
        
    }
    fclose (record_file_cases);
    
    end_timer(-2);
    
    //////////////////////////////////////////////////////////////////////
    // Third method: Select NCASES with the best value of the Q[s][a].
    // To do this, we have to sample NCASES * A_LARGE_NUMBER, to be able
    // to find out the best Q of all of them.
    //////////////////////////////////////////////////////////////////////
    
    float best_mcar_position, best_mcar_velocity, best_Q;
    int best_action;
    
    
    if ( (record_file_cases=fopen("cases-bestQ.txt","w")) == NULL)
    {
        printf("Could not open file for output\nBailing out!\n");
        exit (EXIT_FAILURE);
    }
    
    init_timer();
    
    for (int supersample=0; supersample < 1000; supersample ++)
    {
        best_Q = -10000;
        
        for (int n = 0; n < NCASES; n++)
        {
            // Randomly select state
            mcar_position = dist_pos(eng);
            mcar_velocity = dist_vel(eng);
            
            LoadF();        // Compute feature sets for current car state
            LoadQ();        // Compute all the action values from current F and theta
            
            // Selects the best action
            action = argmax(Q);
            
            if (Q[action] > best_Q)
            {
                best_mcar_position = mcar_position;
                best_mcar_velocity = mcar_velocity;
                best_action = action;
                best_Q = Q[action];
            }
            
            
        }
        
        // save S,A,Q in file.
        fprintf(record_file_cases, "%f %f %d %f\n", best_mcar_position, best_mcar_velocity, best_action, best_Q);

    }
    
    
    fclose (record_file_cases);
    
    end_timer(-3);
    
    
    //////////////////////////////////////////////////////////////////////
    // Fourth method: select NCASES cases from random sampling the state and finding the worst action for that state.
    //////////////////////////////////////////////////////////////////////
    
    if ( (record_file_cases=fopen("cases-argmin.txt","w")) == NULL)
    {
        printf("Could not open file for output\nBailing out!\n");
        exit (EXIT_FAILURE);
    }
    
    
    init_timer();
    for (int n = 0; n < NCASES; n++)
    {
        // Randomly select state
        mcar_position = dist_pos(eng);
        mcar_velocity = dist_vel(eng);
        
        LoadF();        // Compute feature sets for current car state
        LoadQ();        // Compute all the action values from current F and theta
        
        // Selects the best action
        action = argmin(Q);
        
        // save S,A,Q in file.
        fprintf(record_file_cases, "%f %f %d %f\n", mcar_position, mcar_velocity, action, Q[action]);
        
        
    }
    fclose (record_file_cases);
    
    end_timer(-4);
    

    
    
    // this ENDS the Program. DO NOT REMOVE THIS RETURN
    return 0;
}

