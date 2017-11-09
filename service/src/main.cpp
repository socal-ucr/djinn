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
#include <errno.h>
#include <map>
#include <glog/logging.h>
#include <errno.h>
#include <nvml.h>
#include  <signal.h>
#include <limits.h>


#include "boost/program_options.hpp"
#include "socket.h"
#include "thread.h"
#include "tonic.h"

using namespace std;
namespace po = boost::program_options;

map<string, Net<float>*> nets;
int avg=0, peak=0, minimum=INT_MAX;
bool debug;
bool gpu;


void  INThandler(int sig)
{

    signal(sig, SIG_IGN);
    printf("CLEAN UP\n");
	printf("Average = %d W\nPeak = %d W\nMin = %d W\n", avg/1000, peak/1000,
            minimum/1000);
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


void *record_power(void* args)
{
    nvmlDevice_t* device = (nvmlDevice_t*)args;
    unsigned int power;
    int n=1;
    nvmlDeviceGetPowerUsage(*device, &power);
    avg = (int)power;
	while(1)
    {
        nvmlDeviceGetPowerUsage(*device, &power);
        avg = avg+(((int)power-avg)/++n);
        if (peak<(int)power) peak=(int)power;
        if (minimum>(int)power) minimum=(int)power;
        usleep(10);
    }
	return 0;
}


FILE * pFile;
pthread_rwlock_t output_rwlock = PTHREAD_RWLOCK_INITIALIZER;

po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "common,c", po::value<string>()->default_value("../common/"),
      "Directory with configs and weights")(
      "portno,p", po::value<int>()->default_value(8080),
      "Port to open DjiNN on")

      ("nets,n", po::value<string>()->default_value("nets.txt"),
       "File with list of network configs (.prototxt/line)")(
          "weights,w", po::value<string>()->default_value("weights/"),
          "Directory containing weights (in common)")

          ("gpu,g", po::value<bool>()->default_value(false), "Use GPU?")(
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

int main(int argc, char* argv[]) {

signal(SIGINT, INThandler);
//initialize nvml
    nvmlReturn_t result;
    unsigned int device_count;
    nvmlDevice_t device;
    nvmlChkError(nvmlInit(), "nvmlInit");

    nvmlChkError(nvmlDeviceGetCount(&device_count), "DeviceGetCount");

    nvmlChkError(nvmlDeviceGetHandleByIndex(device_count-1, &device), "GetHandle");
    
    nvmlChkError(nvmlDeviceSetPersistenceMode(device,NVML_FEATURE_ENABLED),"EnablePersistence");
//    nvmlChkError(nvmlDeviceSetAutoBoostedClocksEnabled(device,NVML_FEATURE_ENABLED),"DisableAutoBoost");
    //get graphics clock info
    unsigned int memClocksMHz[4];
    unsigned int graphicClocksMHz[4][150];
    unsigned int graphicClockCount[4] = {150, 150, 150, 150};
    unsigned int memClockCount = 4;
    nvmlChkError(nvmlDeviceGetSupportedMemoryClocks(device,&memClockCount,memClocksMHz),"GetMemClocks");

    for(int i = 0; i < memClockCount; i++)
    {
        nvmlChkError(nvmlDeviceGetSupportedGraphicsClocks(device,memClocksMHz[i],&graphicClockCount[i],graphicClocksMHz[i]),"GetGraphicsClocks");

    }

   nvmlChkError(nvmlDeviceSetApplicationsClocks(device, memClocksMHz[0], graphicClocksMHz[0][0]),"SetClocks");
  pthread_t t_rp;
  pthread_create(&t_rp, NULL, record_power, &device);


  // Main thread for the server
  // Spawn a new thread for each request
  po::variables_map vm = parse_opts(argc, argv);
  debug = vm["debug"].as<bool>();
  gpu = vm["gpu"].as<bool>();
  Caffe::set_phase(Caffe::TEST);
  if (vm["gpu"].as<bool>())
    Caffe::set_mode(Caffe::GPU);
  else
    Caffe::set_mode(Caffe::CPU);

  // load all models at init
  ifstream file(vm["nets"].as<string>().c_str());
  string net_name;
  while (file >> net_name) {
    string net = vm["common"].as<string>() + "configs/" + net_name;
    Net<float>* temp = new Net<float>(net);
    const std::string name = temp->name();
    nets[name] = temp;
    std::string weights = vm["common"].as<string>() +
                          vm["weights"].as<string>() + name + ".caffemodel";
    nets[name]->CopyTrainedLayersFrom(weights);
  }

  // how many threads to spawn before exiting
  // -1 to stay indefinitely open
  int total_thread_cnt = vm["threadcnt"].as<int>();
  int socketfd = SERVER_init(vm["portno"].as<int>());

  // Listen on socket
  listen(socketfd, 1000);
  LOG(INFO) << "Server is listening for requests on " << vm["portno"].as<int>();


  //Create log file
  pFile = fopen("out.txt","w");
  if(pFile == NULL){
	LOG(INFO) << "Could not create out file";
	return 0;
	}
  // Main Loop
  int thread_cnt = 0;
  while (1) {
    pthread_t new_thread_id;
    int client_sock = accept(socketfd, (sockaddr*)0, (unsigned int*)0);

    if (client_sock == -1) {
        int errsv = errno;

      LOG(ERROR) << "Failed to accept.\n";
      LOG(ERROR) << errsv;
      LOG(ERROR) << thread_cnt;
      break;
    } else
      new_thread_id = request_thread_init(client_sock);

    ++thread_cnt;
    if (thread_cnt == total_thread_cnt) {
      if (pthread_join(new_thread_id, NULL) != 0) {
        LOG(FATAL) << "Failed to join.\n";
      }
      break;
    }
  }
  return 0;
}
