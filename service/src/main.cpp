/*
 *  Copyright (c) 2015, University of Michigan.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

/**
 * @author: Johann Hauswald, Yiping Kang
 * @contact: jahausw@umich.edu, ypkang@umich.edu
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>
#include <glog/logging.h>
#include <errno.h>
#include <nvml.h>
#include <limits.h>
#include <signal.h>
#include <semaphore.h>
#include <time.h>
#include "boost/program_options.hpp"
#include "socket.h"
#include "thread.h"
#include "tonic.h"
#include <sys/types.h>
#include <unistd.h>
#include "shared_mq.hpp"

//P100

using namespace std;
namespace po = boost::program_options;

map<string, Net<float>*> nets;
bool debug;
int gpu;
int TBLimit;
int numColocate;
std::string net_common;
std::string net_filename;
std::string net_weights;

#define TITANX 1
#define P100   2
#define V100   3

#define GPU_ID V100

#if GPU_ID == 1
    #define NUM_F_STATES 19
    unsigned int F_STATES [NUM_F_STATES]= {0, 8, 16, 24, 32, 40, 48, 56, 69, 72, 80, 88, 96, 103, 111, 119, 127, 135, 140};
#elif GPU_ID == 2 
    #define NUM_F_STATES 10
    unsigned int F_STATES [NUM_F_STATES]= {0, 6, 14, 22, 30, 38, 46, 54, 62, 74};
#elif GPU_ID == 3
    #define NUM_F_STATES 15
    unsigned int F_STATES [NUM_F_STATES]= {4,17,32,44,57,71,84,97,111,124,137,151,164,177,186};
#else
#endif
FILE * logFile;
pthread_rwlock_t output_rwlock = PTHREAD_RWLOCK_INITIALIZER;
pthread_mutex_t GPU_mutex= PTHREAD_MUTEX_INITIALIZER;
std::string outfileName;
unsigned int memClocksMHz[4];
unsigned int graphicClocksMHz[4][200];
unsigned int graphicClockCount[4] = {200, 200, 200, 200};
unsigned int memClockCount = 4;

unsigned long START_TIME;
void  INThandler(int sig)
{
    signal(sig, SIG_IGN);
    printf("CLEAN UP\n");
    nvmlShutdown();
    exit(1);
}



const char * convertToComputeModeString(nvmlComputeMode_t mode)
{
     switch (mode)
     {
         case NVML_COMPUTEMODE_DEFAULT:
             return "Default";
         case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
             return "Exclusive_Thread";
         case NVML_COMPUTEMODE_PROHIBITED:
             return "Prohibited";
         case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
             return "Exclusive Process";
         default:
             return "Unknown";
     }
 }

void nvmlChkError(nvmlReturn_t result, const char * error)
{
    if (NVML_SUCCESS != result)
    {
         printf("ERROR @ %s: %s\n", error, nvmlErrorString(result));

        result = nvmlShutdown();
        if (NVML_SUCCESS != result)
            printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
 
        exit(1);
    } 
}

bool kill_power_t = false;
void *record_power(void* args)
{
    nvmlDevice_t* device = (nvmlDevice_t*)args;
 
    unsigned int power;
    unsigned int clock;

    vector<int> freqArray;
    vector<int> powerArray;
    unsigned long n=1;

    while(!kill_power_t)
    {
        //POWER
        nvmlDeviceGetPowerUsage(*device, &power);
        powerArray.push_back((int)power);

        //SM FREQUENCY
        nvmlDeviceGetClockInfo(*device,NVML_CLOCK_SM,&clock);
        freqArray.push_back((int)clock);
        n++;
        usleep(1000);
    }
   
    std::string poweroutfileName = "power_" + outfileName;
    FILE *power_stats = fopen(poweroutfileName.c_str(), "w");
    std::string header = "power,clock\n";
    fwrite(header.c_str(),sizeof(char), header.length(),power_stats);
    for(int i = 0; i < freqArray.size(); i ++)
    {
        std::string power_temp = std::to_string(powerArray[i]);
        std::string clock_temp = std::to_string(freqArray[i]);
        fwrite(power_temp.c_str(),sizeof(char), power_temp.length(),power_stats);
        fwrite(",",sizeof(char), 1,power_stats);
        fwrite(clock_temp.c_str(),sizeof(char), clock_temp.length(),power_stats);
        fwrite("\n",sizeof(char), 1,power_stats);
    }
    fflush(power_stats);
    fclose(power_stats);
    
    return 0;
}


po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "common,c", po::value<string>()->default_value("../common/"),
      "Directory with configs and weights")(
      "portno,p", po::value<int>()->default_value(8080),
      "Port to open DjiNN on")(
      "tbrf,T", po::value<int>()->default_value(-1),
      "Reduce the # of TB by factor of value")(
      "clock,cl", po::value<int>()->default_value(-1),
      "Set Clock state 0-18,-1 for default")(
      "outfile,o", po::value<string>()->default_value("outfile"),
      "Set outfile name *.out")(
      "power,po", po::value<bool>()->default_value(true),
      "Collect Power stats")(
      "colocate,co", po::value<int>()->default_value(1),
      "Num of colacted requests")(
      "framework,f", po::value<int>()->default_value(-1),
      "If running framework/instance ID")

      ("nets,n", po::value<string>()->default_value("nets.txt"),
       "File with list of network configs (.prototxt/line)")(
          "weights,w", po::value<string>()->default_value("weights/"),
          "Directory containing weights (in common)")

          ("gpu,g", po::value<int>()->default_value(-1), "Set GPU device, set -1 if no GPU")(
              "debug,v", po::value<bool>()->default_value(false),
              "Turn on all debug")("threadcnt,t",
                                   po::value<int>()->default_value(-1),
                                   "Number of threads to spawn before exiting "
                                   "the server. (-1 loop forever)");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}
po::variables_map vm;
shmemList *GPUSend;
shmemList *GPURespond;
boost::interprocess::interprocess_mutex *GPUMutex;
boost::interprocess::interprocess_mutex *respondMutex;
boost::interprocess::interprocess_condition  *GPU_cond;
boost::interprocess::interprocess_condition  *respond_cond;
int GPU_Instance() 
{
    debug = vm["debug"].as<bool>();
    TBLimit = vm["tbrf"].as<int>();
    gpu = vm["gpu"].as<int>();
    if (gpu != -1)
    {
        Caffe::SetDevice(gpu);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_tblimit(TBLimit);
    }
    else
        Caffe::set_mode(Caffe::CPU);
    //Setup shared memory
    int InstanceId = vm["framework"].as<int>();
    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only,"MySharedMemory");
 
    GPUSend = segment.find<shmemList>("GPUSend").first;
    GPURespond = segment.find<shmemList>("GPURespond").first;
    GPUMutex = segment.find<boost::interprocess::interprocess_mutex>("GPUMutex").first;
    respondMutex = segment.find<boost::interprocess::interprocess_mutex>("respondMutex").first;
    GPU_cond = segment.find<boost::interprocess::interprocess_condition>("GPU_cond").first;
    respond_cond = segment.find<boost::interprocess::interprocess_condition>("respond_cond").first;   
    numColocate = vm["colocate"].as<int>();
    // Main Loop
    pthread_t* GPU_thread = (pthread_t*)malloc(numColocate * sizeof(pthread_t));
    int error;
    for(int i = 0; i < numColocate; i ++)
    {
        error = pthread_create(&(GPU_thread[i]), NULL, GPU_handler, (void*)(intptr_t)InstanceId);
        if(error != 0)
        {
            LOG(ERROR) << "Failed to create a GPU handler thread.\nERROR:" << error << "\n";
            exit(1);
        }
    } 
    for(int i = 0; i < numColocate; i ++)
        pthread_join(GPU_thread[i],NULL);

    return 0;
}

int framework() {
    //initialize nvml
    nvmlChkError(nvmlInit(), "nvmlInit");   
    // Main thread for the server
    // Spawn a new thread for each request
    debug = vm["debug"].as<bool>();
    TBLimit = vm["tbrf"].as<int>();
    gpu = vm["gpu"].as<int>();
    numColocate = vm["colocate"].as<int>();
    Caffe::SetDevice(gpu);

    //setup NVML
    nvmlReturn_t result;
    unsigned int device_count;
    nvmlDevice_t device;
    unsigned int pciBusID, pciDeviceID,pciDomainID;
    Caffe::getDevicePCIinfo(pciBusID,pciDeviceID,pciDomainID);
    nvmlChkError(nvmlDeviceGetCount(&device_count), "DeviceGetCount");
    for (unsigned int nvmlDeviceIdx= 0; nvmlDeviceIdx < device_count; ++nvmlDeviceIdx)
    {
        nvmlChkError(nvmlDeviceGetHandleByIndex(nvmlDeviceIdx, &device), "GetHandle");

    	nvmlPciInfo_t nvmPCIInfo;
    	nvmlChkError(nvmlDeviceGetPciInfo ( device, &nvmPCIInfo ),"GetPciInfo");
    
    	if ( pciBusID == nvmPCIInfo.bus && 
			 pciDeviceID == nvmPCIInfo.device &&
		     pciDomainID == nvmPCIInfo.domain )
        	break;
    }

    //get graphics clock info
    nvmlChkError(nvmlDeviceGetSupportedMemoryClocks(device,&memClockCount,memClocksMHz),"GetMemClocks");
    for(int i = 0; i < memClockCount; i++)
        nvmlChkError(nvmlDeviceGetSupportedGraphicsClocks(device,memClocksMHz[i],&graphicClockCount[i],graphicClocksMHz[i]),"GetGraphicsClocks");
  	int fState = vm["clock"].as<int>();
  	if (fState != -1)
		nvmlChkError(nvmlDeviceSetApplicationsClocks(device, memClocksMHz[0], graphicClocksMHz[0][F_STATES[fState]]),"SetClocks");
    else
        nvmlDeviceSetAutoBoostedClocksEnabled(device,NVML_FEATURE_ENABLED);

  // how many threads to spawn before exiting
  // -1 to stay indefinitely open
  int total_thread_cnt = vm["threadcnt"].as<int>();
  int socketfd = SERVER_init(vm["portno"].as<int>());
  bool POWER = vm["power"].as<bool>();
  // Listen on socket
  listen(socketfd, 1000);
  LOG(INFO) << "Server is listening for requests on " << vm["portno"].as<int>();

    //Create log file
    outfileName = vm["outfile"].as<string>() + ".out";
    logFile = fopen(outfileName.c_str(),"w");
    if(logFile == NULL)
    {
        LOG(INFO) << "Could not create out file";
	return 0;
    }
    std::string header = "Time,Name,Queue,Reshape,GPU,end\n";
    
    fwrite(header.c_str(),sizeof(char),header.length(),logFile);
    fflush(logFile);
    
    pthread_t t_rp;
    // Main Loop
    int thread_cnt = 0;

    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC,&time);
    START_TIME = (time.tv_sec * 1000000ul) + (time.tv_nsec/1000ul);

    std::vector<pthread_t> threads;
    
    pthread_t response_thread;
    int error = pthread_create(&response_thread, NULL, response_handler, NULL);
    if(error != 0)
    {
        LOG(ERROR) << "Failed to create a response handler thread.\nERROR:" << error << "\n";
        exit(1);
    }

    //construct Shared memory
    struct shm_remove
    {
        shm_remove() {boost::interprocess::shared_memory_object::remove("MySharedMemory");}
        ~shm_remove(){boost::interprocess::shared_memory_object::remove("MySharedMemory");}
    } remover;

    boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only,"MySharedMemory",4294967296);
    const ShmemAllocator alloc_inst (segment.get_segment_manager());

    GPUSend = segment.construct<shmemList>("GPUSend")[MAX_INSTANCES](alloc_inst);
    GPURespond = segment.construct<shmemList>("GPURespond")[MAX_INSTANCES](alloc_inst);
    GPUMutex = segment.construct<boost::interprocess::interprocess_mutex>("GPUMutex")[MAX_INSTANCES]();
    respondMutex = segment.construct<boost::interprocess::interprocess_mutex>("respondMutex")[MAX_INSTANCES]();
    GPU_cond = segment.construct<boost::interprocess::interprocess_condition>("GPU_cond")[MAX_INSTANCES]();
    respond_cond = segment.construct<boost::interprocess::interprocess_condition>("respond_cond")[MAX_INSTANCES]();

    while (1) 
    {
        pthread_t new_thread_id;
        int client_sock = accept(socketfd, (sockaddr*)0, (unsigned int*)0);

        if (client_sock == -1)
        {
            int errsv = errno;

            LOG(ERROR) << "Failed to accept.\n";
            LOG(ERROR) << errsv;
            LOG(ERROR) << thread_cnt;
            break;
        }
        else
        {
            new_thread_id = request_thread_init(client_sock);
            threads.push_back(new_thread_id);
            if(thread_cnt == 0 && POWER)
                pthread_create(&t_rp, NULL, record_power, &device);
            ++thread_cnt;
        }

        if (thread_cnt == total_thread_cnt)
        {
            for(int i = 0; i < thread_cnt; i++)
            if (pthread_join(threads[i], NULL) != 0)
                LOG(FATAL) << "Failed to join.\n";
            break;
        }
    }

    pthread_cancel(response_thread);
    if(POWER)
    {
        kill_power_t = true;
        pthread_join(t_rp,NULL);
    }
    fclose(logFile);
    nvmlShutdown();
    segment.destroy<shmemList>("GPUSend");
    segment.destroy<shmemList>("GPURespond");
    segment.destroy<boost::interprocess::interprocess_mutex>("GPUMutex");
    segment.destroy<boost::interprocess::interprocess_mutex>("respondMutex");
    segment.destroy<boost::interprocess::interprocess_condition>("GPU_cond");
    segment.destroy<boost::interprocess::interprocess_condition>("respond_cond");


    return 0;
}


int main(int argc, char* argv[]) {
    signal(SIGINT, INThandler);
    vm = parse_opts(argc, argv);

    // load all models at init
    net_common   = vm["common"].as<string>();
    net_filename = vm["nets"].as<string>();
    net_weights  = vm["weights"].as<string>();
    ifstream file(net_filename.c_str());
    string net_name;
    while (file >> net_name)
    {
        string net =  net_common + "configs/" + net_name;
        Net<float>* temp = new Net<float>(net,caffe::TEST);
        const std::string name = temp->name();
        nets[name] = temp;
        std::string weights = net_common + net_weights + name + ".caffemodel";
        nets[name]->CopyTrainedLayersFrom(weights);
    }

    if(vm["framework"].as<int>() == -1)
        framework();
    else
        GPU_Instance();

    return 0;
}
