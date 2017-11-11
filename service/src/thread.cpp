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
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <stdint.h>
#include <ctime>
#include <cmath>
#include <glog/logging.h>
#include <stdio.h>
#include <boost/chrono/thread_clock.hpp>
#include <string>
#include  <signal.h>

#include "timer.h"
#include "thread.h"


extern map<string, Net<float>*> nets;
extern bool debug;
extern bool gpu;
extern FILE * pFile;
extern pthread_rwlock_t output_rwlock;

extern int power_avg, power_peak;
extern int clock_avg, clock_peak;
extern bool reset_stats;


void SERVICE_fwd(float* in, int in_size, float* out, int out_size,
                 Net<float>* net) {

struct timeval t1, t2;
  string net_name = net->name();
  STATS_INIT("service", "DjiNN service inference");
  PRINT_STAT_STRING("network", net_name.c_str());

  if (Caffe::mode() == Caffe::CPU)
    PRINT_STAT_STRING("platform", "cpu");
  else
    PRINT_STAT_STRING("platform", "gpu");

  float loss;
  vector<Blob<float>*> in_blobs = net->input_blobs();

  //tic();

    gettimeofday(&t1, NULL);
  in_blobs[0]->set_cpu_data(in);
  vector<Blob<float>*> out_blobs = net->ForwardPrefilled(&loss);
  memcpy(out, out_blobs[0]->cpu_data(), sizeof(float));
	
  //double time = toc();

   gettimeofday(&t2, NULL);
    double elapsedTime;
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

  PRINT_STAT_DOUBLE("inference latency", elapsedTime);


  STATS_END();


  std::string temp = std::to_string(elapsedTime);

  //write to outputfile
  pthread_rwlock_wrlock(&output_rwlock);
  fwrite(temp.c_str(),sizeof(char),temp.length(), pFile);
  fwrite("\n", sizeof(char), 1, pFile);
  fflush(pFile);
  pthread_rwlock_unlock(&output_rwlock);

  if (out_size != out_blobs[0]->count())
    LOG(FATAL) << "out_size =! out_blobs[0]->count())";
  else
    memcpy(out, out_blobs[0]->cpu_data(), out_size * sizeof(float));
}

pthread_t request_thread_init(int sock) {
  // Prepare to create a new pthread
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1024 * 1024);

  // Create a new thread starting with the function request_handler
  pthread_t tid;
  if (pthread_create(&tid, &attr, request_handler, (void*)(intptr_t)sock) != 0)
    LOG(ERROR) << "Failed to create a request handler thread.\n";

  return tid;
}

void* request_handler(void* sock) {

  int socknum = (intptr_t)sock;

  // 1. Client sends the application type
  // 2. Client sends the size of incoming data
  // 3. Client sends data

  char req_name[MAX_REQ_SIZE];
  SOCKET_receive(socknum, (char*)&req_name, MAX_REQ_SIZE, debug);
  printf("Checking if djinn has to be reset \n");
  //LOG(ERROR) << "Checking if Djinn has to be reset " << req_name << " " << (char*)req_name << endl;
  if (strcmp(req_name, "DONE") == 0)
  {
        char *ok;
        ok= (char*)"OK";
        printf("Done signal received\n");
        SOCKET_send(socknum, ok, MAX_REQ_SIZE, 0);
        LOG(ERROR) << "Reset flag received. Time to reset djinn!!" << endl;
        FILE *power_stats = fopen("power_stats.out", "a");
        if (power_stats == NULL)
        {
            LOG(INFO) << "Cannot create stats file" << endl;
            raise(2);
        }
        std::string power_temp_avg   = std::to_string(power_avg/1000);
        std::string power_temp_peak  = std::to_string(power_peak/1000);
        std::string clock_temp_avg   = std::to_string(clock_avg);
        std::string clock_temp_peak  = std::to_string(clock_peak);
        
        //POWER
        fwrite(power_temp_avg.c_str(), sizeof(char), power_temp_avg.length(), power_stats);
        fwrite(",", sizeof(char), 1, power_stats);
        fwrite(power_temp_peak.c_str(), sizeof(char), power_temp_peak.length(), power_stats);
        //SM FREQUENCY
        fwrite(clock_temp_avg.c_str(), sizeof(char), clock_temp_avg.length(), power_stats);
        fwrite(",", sizeof(char), 1, power_stats);
        fwrite(clock_temp_peak.c_str(), sizeof(char), clock_temp_peak.length(), power_stats);

        fwrite("\n", sizeof(char), 1, power_stats);
        fflush(power_stats);
        fclose(power_stats);
        reset_stats = true;
  }
  map<string, Net<float>*>::iterator it = nets.find(req_name);
  if (it == nets.end()) {
    LOG(ERROR) << "Task " << req_name << " not found.";
    return (void*)1;
  } else
    LOG(INFO) << "Task " << req_name << " forward pass.";

  // receive the input data length (in float)
  int sock_elts = SOCKET_rxsize(socknum);
  if (sock_elts < 0) {
    LOG(ERROR) << "Error num incoming elts.";
    return (void*)1;
  }

  // reshape input dims if incoming data != current net config
  LOG(INFO) << "Elements received on socket " << sock_elts << endl;

  reshape(nets[req_name], sock_elts);

  int in_elts = nets[req_name]->input_blobs()[0]->count();
  int out_elts = nets[req_name]->output_blobs()[0]->count();
  float* in = (float*)malloc(in_elts * sizeof(float));
  float* out = (float*)malloc(out_elts * sizeof(float));

  // Main loop of the thread, following this order
  // 1. Receive input feature (has to be in the size of sock_elts)
  // 2. Do forward pass
  // 3. Send back the result
  // 4. Repeat 1-3

  // Warmup: used to move: the network to the device for the first time
  // In all subsequent forward passes, the trained model resides on the
  // device (GPU)
  bool warmup = true;

  while (1) {
    LOG(INFO) << "Reading from socket.";
    int rcvd =
        SOCKET_receive(socknum, (char*)in, in_elts * sizeof(float), debug);

    if (rcvd == 0) break;  // Client closed the socket

    if (warmup && gpu) {
      float loss;
        printf("INPUT_BLOBS\n");
      vector<Blob<float>*> in_blobs = nets[req_name]->input_blobs();
        printf("SET_CPU_DATA\n");
      in_blobs[0]->set_cpu_data(in);
        printf("ForwardPrefilled\n");
      vector<Blob<float>*> out_blobs;
      out_blobs = nets[req_name]->ForwardPrefilled(&loss);
        printf("END\n");
      warmup = false;
    }

    LOG(INFO) << "Executing forward pass.";
    SERVICE_fwd(in, in_elts, out, out_elts, nets[req_name]);

    LOG(INFO) << "Writing to socket.";
    SOCKET_send(socknum, (char*)out, out_elts * sizeof(float), debug);
  }

  // Exit the thread
  LOG(INFO) << "Socket closed by the client.";

  free(in);
  free(out);
  pthread_exit((void*)0);
  return (void*)0;
}
