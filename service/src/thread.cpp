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
#include <cuda.h>
#include <semaphore.h>
#include <errno.h>
#include <queue>
#include "timer.h"
#include "thread.h"
#include "cu_helper.h"
#include <map>

map<pthread_t,int> condMap;
extern map<string, Net<float>*> nets;
extern bool debug;
extern bool gpu;
extern FILE * pFile;
extern pthread_rwlock_t output_rwlock;
extern sem_t CUDA_sem;
extern pthread_mutex_t GPU_mutex;
extern pthread_mutex_t active_thread_mutex;
extern pthread_mutex_t queue_mutex1;
extern pthread_mutex_t queue_mutex2;

extern int power_avg, power_peak;
extern int clock_avg, clock_peak;
extern bool reset_stats;
extern int active_threads;
extern unsigned int current_requests;

extern pthread_cond_t * condArray;
extern std::queue<pthread_cond_t*> thread_queue[NUM_QUEUES];
bool FLAG = false;
extern unsigned long START_TIME;

unsigned long SERVICE_fwd(float* in, int in_size, float* out, int out_size,
                 Net<float>* net,unsigned long &qtime)
{

    struct timespec st1, st2,qt1,qt2;
    string net_name = net->name();
    //STATS_INIT("service", "DjiNN service inference");
    //  PRINT_STAT_STRING("network", net_name.c_str());

    //  if (Caffe::mode() == Caffe::CPU)
    //    PRINT_STAT_STRING("platform", "cpu");
    //  else
    //    PRINT_STAT_STRING("platform", "gpu");

    float loss;
    bool empty;
    pthread_cond_t* cond = &(condArray[condMap[pthread_self()]]);
    int queueID = condMap[pthread_self()] % NUM_QUEUES;

    pthread_mutex_t* queue_mutex;
    if (queueID == 0)
        queue_mutex = &queue_mutex1;
    else
        queue_mutex = &queue_mutex2;

    
    pthread_mutex_lock(queue_mutex);
    thread_queue[queueID].push(cond);
    current_requests++;
    clock_gettime(CLOCK_MONOTONIC,&qt1);
    while(thread_queue[queueID].front() != cond)
    {
        pthread_cond_wait(cond,queue_mutex);
    }
    clock_gettime(CLOCK_MONOTONIC,&qt2);
    pthread_mutex_unlock(queue_mutex);
    qtime = (qt2.tv_sec - qt1.tv_sec) * 1000000ul;
    qtime += (qt2.tv_nsec - qt1.tv_nsec) / 1000;

    vector<Blob<float>*> in_blobs = net->input_blobs();

    clock_gettime(CLOCK_MONOTONIC,&st1);

    in_blobs[0]->set_cpu_data(in);
    vector<Blob<float>*> out_blobs = net->ForwardPrefilled(&loss);
    memcpy(out, out_blobs[0]->cpu_data(), sizeof(float));

    clock_gettime(CLOCK_MONOTONIC,&st2);

    pthread_mutex_lock(queue_mutex);
    thread_queue[queueID].pop();
    //current_requests--;
    if(!thread_queue[queueID].empty())
    {
        pthread_cond_t* next = thread_queue[queueID].front();
        int ret = pthread_cond_signal(next);
    }

    pthread_mutex_unlock(queue_mutex);

    unsigned long elapsedTime = (st2.tv_sec - st1.tv_sec) * 1000000;
    elapsedTime += (st2.tv_nsec - st1.tv_nsec) / 1000;

    // PRINT_STAT_DOUBLE("inference latency", elapsedTime);
    // STATS_END();

    if (out_size != out_blobs[0]->count())
        LOG(FATAL) << "out_size =! out_blobs[0]->count())";
    else
        memcpy(out, out_blobs[0]->cpu_data(), out_size * sizeof(float));

    return elapsedTime;
}

pthread_t request_thread_init(int sock) 
{
    // Prepare to create a new pthread
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

    // Create a new thread starting with the function request_handler
    pthread_t tid;
    pthread_mutex_lock(&active_thread_mutex);
    int error = pthread_create(&tid, &attr, request_handler, (void*)(intptr_t)sock);
    if(error != 0)
        LOG(ERROR) << "Failed to create a request handler thread.\nERROR:" << error << "\n";

    condMap.insert(std::pair<pthread_t,int>(tid,active_threads));
    active_threads++;
    pthread_mutex_unlock(&active_thread_mutex);

    return tid;
}

void* request_handler(void* sock)
{
    int socknum = (intptr_t)sock;

    // 1. Client sends the application type
    // 2. Client sends the size of incoming data
    // 3. Client sends data
    
    std::vector<std::string> output;
    char req_name[MAX_REQ_SIZE];
    SOCKET_receive(socknum, (char*)&req_name, MAX_REQ_SIZE, debug);
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
        fwrite(",", sizeof(char), 1, power_stats);
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
    if (it == nets.end())
    {
        LOG(ERROR) << "Task " << req_name << " not found.";
        return (void*)1;
    }
    else
        LOG(INFO) << "Task " << req_name << " forward pass.";

    // receive the input data length (in float)
     int sock_elts = SOCKET_rxsize(socknum);
    if (sock_elts < 0)
    {
        LOG(ERROR) << "Error num incoming elts.";
        return (void*)1;
    }

    // reshape input dims if incoming data != current net config
    // LOG(INFO) << "Elements received on socket " << sock_elts << endl;

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


    struct timespec t1, t2,time;
    while (1) 
    {
        
        clock_gettime(CLOCK_MONOTONIC,&t1);
        // LOG(INFO) << "Reading from socket.";
        int rcvd =
            SOCKET_receive(socknum, (char*)in, in_elts * sizeof(float), debug);

        clock_gettime(CLOCK_MONOTONIC,&time);
        unsigned long stamp = (time.tv_sec * 1000000ul) + (time.tv_nsec/1000ul);
        stamp -= START_TIME;
        if (rcvd == 0) break;  // Client closed the socket


        //LOG(INFO) << "Executing forward pass.";
        unsigned long qtime;
        unsigned long service_time = SERVICE_fwd(in, in_elts, out, out_elts, nets[req_name],qtime);

       // LOG(INFO) << "Writing to socket.";
       //SOCKET_send(socknum, (char*)out, out_elts * sizeof(float), debug);

        //write to outputfile
        
        clock_gettime(CLOCK_MONOTONIC,&t2);
        unsigned long total = (t2.tv_sec - t1.tv_sec) * 1000000ul;
        total += (t2.tv_nsec - t1.tv_nsec) / 1000ul;
        output.push_back(std::to_string(stamp) + "," + std::to_string(service_time) + "," + std::to_string(qtime) + "," + std::to_string(total) + "\n");

    }

    // Exit the thread
    LOG(INFO) << "Socket closed by the client.";
    for (int i = 0; i < output.size(); i ++)
    {
        pthread_rwlock_wrlock(&output_rwlock);
        fwrite(output[i].c_str(),sizeof(char),output[i].length(), pFile);
        fflush(pFile);
        pthread_rwlock_unlock(&output_rwlock);
    }

    free(in);
    free(out);
    pthread_mutex_lock(&active_thread_mutex);
    active_threads--;
    printf("CLOSE THREAD%d\n",active_threads);
    pthread_mutex_unlock(&active_thread_mutex);
   // pthread_detach(pthread_self());
    return (void*)0;
}
