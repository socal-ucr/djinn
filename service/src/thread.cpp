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
#include <signal.h>
#include <cuda.h>
#include <semaphore.h>
#include <errno.h>
#include "timer.h"
#include "thread.h"
#include <map>
#include <list>

struct Request
{
    unsigned long time;
    int socknum;
    char req_name[MAX_REQ_SIZE];
    int sock_elts;
    int in_elts;
    int out_elts;
    float* in;
    float* out;
};

map<pthread_t,int> condMap;
extern map<string, Net<float>*> nets;
extern bool debug;
extern FILE * pFile;
extern pthread_rwlock_t output_rwlock;

extern int power_avg, power_peak;
extern int clock_avg, clock_peak;
extern bool reset_stats;

pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t GPU_handle_cond = PTHREAD_COND_INITIALIZER;
std::list<Request> GPU_queue;
extern unsigned long START_TIME;

//unsigned long SERVICE_fwd(float* in, int in_size, float* out, int out_size,
//                Net<float>* net,unsigned long &qtime)
void * GPU_handler(void * args)
{
    struct timespec tStart, tEnd;
    unsigned long queueTime, gpuTime,reshapeTime;
    while(1)
    {
        pthread_mutex_lock(&queue_mutex);
        
        while(GPU_queue.empty())
            pthread_cond_wait(&GPU_handle_cond,&queue_mutex);

        //get data from queue
        Request current_req = GPU_queue.front();
        GPU_queue.pop_front();
        pthread_mutex_unlock(&queue_mutex);
        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        queueTime = ((tEnd.tv_sec * 1000000ul) + tEnd.tv_nsec / 1000ul) - current_req.time;

        //reshape net for current input
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Net<float>* net = nets[current_req.req_name]; 

        reshape(net, current_req.sock_elts);
        vector<Blob<float>*> in_blobs = net->input_blobs();
        clock_gettime(CLOCK_MONOTONIC,&tEnd);

        reshapeTime = (tEnd.tv_sec - tStart.tv_sec) * 1000000ul;
        reshapeTime += (tEnd.tv_nsec - tStart.tv_nsec) / 1000ul;

        //Send to GPU
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        float loss;
        in_blobs[0]->set_cpu_data(current_req.in);
        vector<Blob<float>*> out_blobs = net->ForwardPrefilled(&loss);
        memcpy(current_req.out, out_blobs[0]->cpu_data(), sizeof(float));

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        gpuTime = (tEnd.tv_sec - tStart.tv_sec) * 1000000ul;
        gpuTime += (tEnd.tv_nsec - tStart.tv_nsec) / 1000ul;

        if (current_req.out_elts != out_blobs[0]->count())
            LOG(FATAL) << "out_size =! out_blobs[0]->count())";
        else
            memcpy(current_req.out, out_blobs[0]->cpu_data(), current_req.out_elts * sizeof(float));

       SOCKET_send(current_req.socknum, (char*)current_req.out, current_req.out_elts * sizeof(float), debug); 
    }
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
    int error = pthread_create(&tid, &attr, request_handler, (void*)(intptr_t)sock);
    if(error != 0)
        LOG(ERROR) << "Failed to create a request handler thread.\nERROR:" << error << "\n";

    return tid;
}


void* request_handler(void* sock)
{
    int socknum = (intptr_t)sock;

    // 1. Client sends the application type
    // 2. Client sends the size of incoming data
    // 3. Client sends data
    
    struct timespec time;
    while (1) 
    {
        struct Request req;
        //Receive Type of request
        SOCKET_receive(socknum, (char*)&req.req_name, MAX_REQ_SIZE, debug);

        //check if valid request type
        map<string, Net<float>*>::iterator it = nets.find(req.req_name);
        if (it == nets.end())
        {
            LOG(ERROR) << "Task " << req.req_name << " not found.";
            return (void*)1;
        }
        else
            LOG(INFO) << "Task " << req.req_name << " forward pass.";
        
        req.socknum = socknum;
        //Receive the input data length (in float)
        req.sock_elts = SOCKET_rxsize(socknum);
        if (req.sock_elts < 0)
        {
            LOG(ERROR) << "Error num incoming elts.";
            return (void*)1;
        }
    
        //calculate size to allocate memory
        int in_elts, out_elts;
        calculateShape(nets[req.req_name],req.sock_elts,in_elts,out_elts);
        req.in_elts = in_elts;
        req.in = (float*)malloc(in_elts * sizeof(float));
        req.out_elts = out_elts;
        req.out = (float*)malloc(out_elts * sizeof(float));

        //Recieve Data
        int rcvd =
            SOCKET_receive(socknum, (char*)req.in, in_elts * sizeof(float), debug);
        if (rcvd == 0) break;  // Client closed the socket
        clock_gettime(CLOCK_MONOTONIC,&time);
        req.time = ((time.tv_sec * 1000000ul) + (time.tv_nsec/1000ul)) - START_TIME;

        //Add to queue
        pthread_mutex_lock(&queue_mutex);
        GPU_queue.push_back(req);
        pthread_mutex_unlock(&queue_mutex);

        free(req.in);
        free(req.out);
       //SOCKET_send(socknum, (char*)out, out_elts * sizeof(float), debug);
    }

    // Exit the thread
    LOG(INFO) << "Socket closed by the client.";
   // pthread_detach(pthread_self());
    return (void*)0;
}

void * send_handler()
{

}
