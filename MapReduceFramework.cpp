#include <pthread.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <atomic>
#include <unistd.h>
#include "MapReduceFramework.h"
#include "Barrier.h"

#define M_LOCK_SYS_ERR_MSG "system error: lock mutex failed\n"
#define M_UNLOCK_SYS_ERR_MSG "system error: unlock mutex failed\n"
#define M_INIT_SYS_ERR_MSG "system error: init mutex failed\n"
#define M_DESTROY_SYS_ERR_MSG "system error: destroy mutex failed\n"
#define T_CREATE_SYS_ERR_MSG "system error: create thread failed\n"
#define T_JOIN_SYS_ERR_MSG "system error: join thread failed\n"



#define INITIAL_PERCENTAGE 0
#define INITIAL_ATOMIC_VALUE 0

using namespace std;

/** TYPES DECLARATIONS **/
class JobContext;
typedef pair<pthread_t, IntermediateVec> ThreadPair;
typedef vector<IntermediateVec> VectorOfInterVectors;


/** FUNCTION DECLARATIONS **/
void *threadRoutine(void *args);
void mutex_lock(pthread_mutex_t *mutex);
void mutex_unlock(pthread_mutex_t *mutex);
void percentageUpdater(JobContext *job_context);
void initJobState(stage_t stage, JobContext *job_context);
void mutex_destroy(pthread_mutex_t *mutex);

/**JOB CONTEXT STRUCT **/

class JobContext {
public:
    const MapReduceClient *client;
    const InputVec *inputVec; //V1,K1
    OutputVec *outputVec;//V3,K3
    vector<pthread_t> threadIds;
    map<pthread_t, IntermediateVec> *threadsMap{}; // ThreadID: current vector we are working on - v2,k3
    atomic<unsigned long> * elementsFoundInInputVec;
    atomic<unsigned long> elementsMapped{};
    atomic<unsigned long> elementsToShuffle{};
    atomic<unsigned long> elementsShuffled{};
    atomic<unsigned long> elementsFoundInInterVec{};
    atomic<unsigned long> elementsReduced{};
    atomic<unsigned long> waitFlag{};
    VectorOfInterVectors vectorOfInterVectors; //vectors of (k2,v2) before reduce
    Barrier *barrier;
    JobState job_state{}; // current state of the job -> state+ percentage
    pthread_mutex_t threadsMapMutex;
    pthread_mutex_t emit3Mutex;
    pthread_mutex_t stateHandlerMutex;


    /**
     * Constructor for the JobContext class
     * @param client
     * @param inputVec
     * @param outputVec
     * @param multiThreadLevel
     */
    JobContext(const MapReduceClient &client, const InputVec &inputVec,
               OutputVec &outputVec, int multiThreadLevel) :
            threadIds(multiThreadLevel,0),
            waitFlag(false),
            threadsMapMutex((pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER),
            emit3Mutex((pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER),
            stateHandlerMutex((pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER)
   {
        this->client = &client;
        this->inputVec = &inputVec;
        this->outputVec = &outputVec;
        this->barrier = new Barrier(multiThreadLevel);
        this->threadsMap = new map<pthread_t, IntermediateVec>;
        this->elementsFoundInInputVec = new std::atomic<unsigned long>(0);
        this->elementsMapped = INITIAL_ATOMIC_VALUE;
        this->elementsToShuffle = INITIAL_ATOMIC_VALUE;
        this->elementsShuffled = INITIAL_ATOMIC_VALUE;
        this->elementsReduced = INITIAL_ATOMIC_VALUE;
        this->elementsFoundInInterVec = INITIAL_ATOMIC_VALUE;
        this->waitFlag = INITIAL_ATOMIC_VALUE;
        this->job_state = {MAP_STAGE,INITIAL_PERCENTAGE};


        //create all threads
        for (int i = 0; i < multiThreadLevel; i++) {
            pthread_t pid;
            IntermediateVec interVec;
            if (pthread_create(&this->threadIds.at(i), nullptr,
                               threadRoutine, this) != EXIT_SUCCESS) {
                fprintf(stderr, T_CREATE_SYS_ERR_MSG);
                exit(EXIT_FAILURE);
            }
//            ThreadPair newThreadPair = ThreadPair(pid, interVec);
        }
   }

    /**
     * Destructor for job context
     */
    ~JobContext()
    {
        mutex_destroy(&threadsMapMutex);
        mutex_destroy(&emit3Mutex);
        mutex_destroy(&stateHandlerMutex);
        delete threadsMap;
        delete elementsFoundInInputVec;
        delete barrier;

    }
};

/**
 * Wrapper dor pthread mutex lock with error messages
 * @param mutex
 */
void mutex_lock(pthread_mutex_t *mutex) {
    if (pthread_mutex_lock(mutex) != EXIT_SUCCESS) {
        fprintf(stderr, M_LOCK_SYS_ERR_MSG);
        exit(EXIT_FAILURE);
    }
}

/**
 * Wrapper dor pthread mutex unlock with error messages
 * @param mutex
 */
void mutex_unlock(pthread_mutex_t *mutex) {
    if (pthread_mutex_unlock(mutex) != EXIT_SUCCESS) {
        fprintf(stderr, M_UNLOCK_SYS_ERR_MSG);
        exit(EXIT_FAILURE);
    }
}

/**
 * Wrapper dor pthread mutex destroy with error messages
 * @param mutex
 */
void mutex_destroy(pthread_mutex_t *mutex) {
    if (pthread_mutex_destroy(mutex) != EXIT_SUCCESS) {
        fprintf(stderr, M_DESTROY_SYS_ERR_MSG);
        exit(EXIT_FAILURE);
    }
}

bool compareIntermediatePairs(IntermediatePair p1, IntermediatePair p2) {
    return *(p1.first) < *(p2.first);
}


/**
 * This function is part of thread_routine.
 * Every thread will reach this part and will try to take as much inputPairs
 * as he can until a context switch occurs. The client's map implementation will
 * perform the mapping on the inputPair and add it to a unique IntermediatePair
 * using the emit2 function.
 * @param pid
 * @param jobContext
 */
void mapStage(JobContext *jobContext) {
    auto inputVecSize = jobContext->inputVec->size();
    auto old_value = (*jobContext->elementsFoundInInputVec)++;
    while (old_value < inputVecSize) {
        InputPair inputPair = jobContext->inputVec->at(old_value);
        jobContext->client->map(inputPair.first, inputPair.second, jobContext);
        mutex_lock(&jobContext->stateHandlerMutex);
        percentageUpdater(jobContext);
        mutex_unlock(&jobContext->stateHandlerMutex);
        old_value = (*jobContext->elementsFoundInInputVec)++;
    }
}

/**
 * This function is part of thread_routine.
 * Every thread will reach this part and will sort his Intermediate Vector
 * using compareIntermediatePairs.
 * @param jobContext
 */
void sortStage(JobContext *jobContext) {
    pthread_t pid = pthread_self();
    mutex_lock(&jobContext->threadsMapMutex);
    sort((jobContext->threadsMap)->operator[](pid).begin(),
         (jobContext->threadsMap)->operator[](pid).end(),
         compareIntermediatePairs);
    mutex_unlock(&jobContext->threadsMapMutex);

}

/**
 * This function is part of thread_routine.
 * Every thread will reach this part and only the first created thread will actually
 * perform the shuffle process.
 * The shuffle process sorts all pairs by their keys to a unique vector.
 * @param jobContext
 */
void shuffleStage(JobContext *jobContext) {
    pthread_t pid = pthread_self();
    if (pid != jobContext->threadIds.at(0)) {
        return;
    }
    initJobState(SHUFFLE_STAGE, jobContext);
    while (jobContext->elementsShuffled < jobContext->elementsToShuffle) {
        IntermediatePair *maxKeyPair = nullptr;
        // first loop to find the greatest key among the sorted vectors.
        for (auto &it: *jobContext->threadsMap) {
            if (it.second.empty())
                continue;
            else if ((maxKeyPair == nullptr) || (compareIntermediatePairs(*maxKeyPair, it.second.back())))
                maxKeyPair = &it.second.back();
        }
        IntermediateVec currentGreatestVector;
        for (auto &it: *jobContext->threadsMap) {
            while ((!it.second.empty())
                    && (jobContext->elementsShuffled < jobContext->elementsToShuffle)
                    && (!compareIntermediatePairs(it.second.back(), *maxKeyPair))) {
                currentGreatestVector.push_back(it.second.back());
                it.second.pop_back();
                percentageUpdater(jobContext);
            }

        }
        jobContext->vectorOfInterVectors.push_back(currentGreatestVector);
    }
}

/**
 * This function is part of thread_routine.
 * Every thread will reach this part and will try to take sny available keys
 * to perform the reduce function on them. The client's reduce implementation will
 * perform the mapping on the IntermediateVector and add it to the OutputVector
 * using the emit3 function.
 * @param jobContext
 */
void reduceStage(JobContext *job_context) {
    auto old_value = job_context->elementsFoundInInterVec++;

    while (old_value < job_context->vectorOfInterVectors.size()) {
        IntermediateVec pairs = job_context->vectorOfInterVectors.at(old_value);
        job_context->client->reduce(&pairs, job_context);
        mutex_lock(&job_context->stateHandlerMutex);
        percentageUpdater(job_context);
        mutex_unlock(&job_context->stateHandlerMutex);
        old_value = job_context->elementsFoundInInterVec++;
    }

}


/**
 * Runs the Map, Shuffle and Reduce stages for each thread.
 * @param args - given as the forth parameter in pthread_create function call.
 */
void *threadRoutine(void *args) {
    auto *job_context = (JobContext *) args;
    mapStage(job_context);
    sortStage(job_context);
    job_context->barrier->barrier();
    shuffleStage(job_context);
    job_context->barrier->barrier();
    initJobState(REDUCE_STAGE,job_context);
    reduceStage(job_context);
    return nullptr;
}

/**
 * This function updates the job state for the given job context
 * @param stage stage to update
 * @param job_context job to be updated
 */
void initJobState(const stage_t stage, JobContext *job_context) {
    mutex_lock(&job_context->stateHandlerMutex);
    if (job_context->job_state.stage != stage) {
        job_context->job_state.stage = stage;
        job_context->job_state.percentage = 0;
    }
    mutex_unlock(&job_context->stateHandlerMutex);
}
/**
 * This function updates the percentage for the given job
 * @param job_context
 */
void percentageUpdater(JobContext *job_context)
{
    unsigned long elementsDone;
    unsigned long totalElements;
    switch (job_context->job_state.stage) {
        case UNDEFINED_STAGE:
            return;
        case MAP_STAGE:
            job_context->elementsMapped++;
            elementsDone = job_context->elementsMapped.load();
            totalElements = job_context->inputVec->size();
            break;
        case SHUFFLE_STAGE:
            job_context->elementsShuffled++;
            elementsDone = job_context->elementsShuffled.load();
            totalElements = job_context->elementsToShuffle.load();
            break;
        case REDUCE_STAGE:
            job_context->elementsReduced++;
            elementsDone = job_context->elementsReduced.load();
            totalElements = job_context->vectorOfInterVectors.size();
            break;
    }
    if (elementsDone <= totalElements) {
        job_context->job_state.percentage = (float)(elementsDone * 100) / (float)totalElements;
    }

}

/** API FUNCTIONS **/

/**
 * The function receives as input intermediary element (K2, V2) and
 * context which contains data structure of the thread that created
 * the intermediary element. The function saves the intermediary element
 * in the context data structures. In addition, the function updates
 * the number of intermediary elements using atomic counter.
 * Please pay attention that emit2 is called from the client's map function
 * and the context is passed from the framework to the client's map function as parameter.
 * @param key
 * @param value
 * @param context
 */
void emit2(K2 *key, V2 *value, void *context) {
    auto *job_context = (JobContext *) context;
    pthread_t pid = pthread_self();
    mutex_lock(&job_context->threadsMapMutex);
    job_context->threadsMap->operator[](pid).push_back(IntermediatePair(key, value));
    job_context->elementsToShuffle++;
    mutex_unlock(&job_context->threadsMapMutex);
}

/**
 * The function receives as input output element (K3, V3) and
 * context which contains data structure of the thread that created the output element.
 * The function saves the output element in the context data structures (output vector).
 * In addition, the function updates the number of output elements using atomic counter.
 * Please pay attention that emit3 is called from the client's map function
 * and the context is passed from the framework to the client's map function as parameter.
 * @param key
 * @param value
 * @param context
 */
void emit3(K3 *key, V3 *value, void *context) {
    auto *job_context = (JobContext *) context;
    mutex_lock(&job_context->emit3Mutex);
    job_context->outputVec->push_back(OutputPair(key,value));
    mutex_unlock(&job_context->emit3Mutex);
}

/**
 * This function starts the mapReduce alg
 * @param client The implementation of MapReduceClient = the task that the framework should run.
 * @param inputVec a vector of type std::vector<std::pair<K1*, V1*>>, the input elements.
 * @param outputVec empty vector of type std::vector<std::pair<K3*, V3*>>,
 * to which the output elements will be added before returning.
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm.
 * @return The function returns JobHandle that will be used for monitoring the job.
 */
JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec,
                            OutputVec &outputVec,
                            int multiThreadLevel) {
    JobHandle job_handle = new JobContext(client, inputVec, outputVec, multiThreadLevel);
    return job_handle;
}

/**
 * This function gets JobHandle returned by startMapReduceJob and waits until it is finished.
 * @param job
 */
void waitForJob(JobHandle job) {
    auto job_context = (JobContext *) job;
    if (job_context->waitFlag.load() > 0)
        return;
    else {
        job_context->waitFlag++;
    }
    for (size_t i=0 ; i < job_context->threadIds.size() ; i++) {
        if (pthread_join(job_context->threadIds[i], nullptr) != EXIT_SUCCESS) {
            fprintf(stderr, T_JOIN_SYS_ERR_MSG);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 *  this function gets a JobHandle and updates the state of the job into the given JobState struct.
 * @param job whose state should be updated
 * @param state to be updated
 */
void getJobState(JobHandle job, JobState *state) {
    auto *job_context = static_cast<JobContext*>(job);
    mutex_lock(&job_context->stateHandlerMutex);
    state->stage = job_context->job_state.stage;
    state->percentage = job_context->job_state.percentage;
    mutex_unlock(&job_context->stateHandlerMutex);

}

/**
 * This function Releases all resources of a job.
 * After this function is called the job handle will be invalid.
 * In case that the function is called and the job is not finished will wait
 * until the job is finished to close it.
 * @param job to be closed
 */
void closeJobHandle(JobHandle job) {
    auto job_context = (JobContext *) job;
    waitForJob(job);
    delete job_context;

}
