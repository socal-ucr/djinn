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
#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp"
#include "caffe/caffe.hpp"
#include "align.h"
#include "socket.h"
#include "tonic.h"
#include <pthread.h>


using namespace std;
using namespace cv;

TonicSuiteApp app;
pthread_t* threads;



FILE * pFile;
pthread_rwlock_t output_rwlock = PTHREAD_RWLOCK_INITIALIZER;

/*
void tic(void) { gettimeofday(&t1, NULL); }

double toc(void) {
   gettimeofday(&t2, NULL);
   double elapsedTime;
   elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
   elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
 
   return elapsedTime;
 }
*/

unsigned int microseconds = 1000000;
namespace po = boost::program_options;
struct timeval tv1, tv2;

unsigned int RPS;
unsigned int seconds;
unsigned int total_requests;
double sleep_time;
void* create_thread_socket(void *args)
{
   int socketfd = CLIENT_init((char*)app.hostname.c_str(), app.portno, 0);

   float* preds = (float*)malloc(app.pl.num * sizeof(float));

   for(int i = 0; i < seconds; i++)
{
    SOCKET_send(socketfd, (char*)&app.pl.req_name, MAX_REQ_SIZE, 0);

    // send len
    SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

    // send image(s)
    SOCKET_send(socketfd, (char*)app.pl.data,
                app.pl.num * app.pl.size * sizeof(float), 0);

    usleep(microseconds);
    //SOCKET_receive(socketfd, (char*)preds, app.pl.num * sizeof(float),
    //               0);
    }
for(int i = 0; i < seconds;i ++)
{
	SOCKET_receive(socketfd, (char*)preds, app.pl.num * sizeof(float),
                   0);
}
    SOCKET_close(app.socketfd, 0);

   free(preds);

    //write to outputfile
/*
    pthread_rwlock_wrlock(&output_rwlock);
    fwrite(temp.c_str(),sizeof(char),temp.length(), pFile);
    fwrite("\n", sizeof(char), 1, pFile);
    fflush(pFile);
    pthread_rwlock_unlock(&output_rwlock);
*/
    pthread_exit(NULL);

}

void reset_djinn()
{
    char recvd[MAX_REQ_SIZE], *done;
    LOG(INFO) << "Sending Signal" << endl;
    done = (char*)"DONE";
    int socketfd = CLIENT_init((char*)app.hostname.c_str(), app.portno, 0);
    SOCKET_send(socketfd, done, MAX_REQ_SIZE, 0);
    LOG(INFO) << "Sending Complete" << endl;
    SOCKET_receive(socketfd, recvd, MAX_REQ_SIZE, 0);
    LOG(INFO) << "Received " << recvd << endl;
    if (strcmp(recvd, "OK") == 0)
        LOG(INFO) << "Finished processing all requests successfully"<<endl;
    SOCKET_close(socketfd, 0);
    exit(1);
}

po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "common,c", po::value<string>()->default_value("../../common/"),
      "Directory with configs and weights")(
      "task,t", po::value<string>()->default_value("imc"),
      "Image task: imc (ImageNet), face (DeepFace), dig (LeNet)")(
      "network,n", po::value<string>()->default_value("imc.prototxt"),
      "Network config file (.prototxt)")(
      "weights,w", po::value<string>()->default_value("imc.caffemodel"),
      "Pretrained weights (.caffemodel)")(
      "input,i", po::value<string>()->default_value("imc-list.txt"),
      "List of input images (1 jpg/line)")(
      "rps,r", po::value<int>()->default_value(1),
      "Requests per Second")(
      "seconds,s", po::value<int>()->default_value(1),
      "Time program will run for in seconds")

      ("djinn,d", po::value<bool>()->default_value(false),
       "Use DjiNN service?")("hostname,o",
                             po::value<string>()->default_value("localhost"),
                             "Server IP addr")(
          "portno,p", po::value<int>()->default_value(8080), "Server port")

      // facial recognition flags
      ("align,l", po::value<bool>()->default_value(true),
       "(face) align images before inference?")(
          "haar,a", po::value<string>()->default_value("data/haar.xml"),
          "(face) Haar Cascade model")(
          "flandmark,f",
          po::value<string>()->default_value("data/flandmark.dat"),
          "(face) Flandmarks trained data")

          ("gpu,g", po::value<bool>()->default_value(false), "Use GPU?")(
              "debug,v", po::value<bool>()->default_value(false),
              "Turn on all debug");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}

int main(int argc, char** argv) {

  po::variables_map vm = parse_opts(argc, argv);

  bool debug = vm["debug"].as<bool>();

  app.task = vm["task"].as<string>();
  app.network =
      vm["common"].as<string>() + "configs/" + vm["network"].as<string>();
  app.weights =
      vm["common"].as<string>() + "weights/" + vm["weights"].as<string>();
  app.input = vm["input"].as<string>();

  RPS = vm["rps"].as<int>();
  seconds = vm["seconds"].as<int>();
  total_requests = RPS * seconds;
  sleep_time = (1000000.0f / double(RPS));

  
  threads = (pthread_t*)malloc(sizeof(pthread_t)*RPS);
  // DjiNN service or local?
  app.djinn = vm["djinn"].as<bool>();
  app.gpu = vm["gpu"].as<bool>();

  if (app.djinn) {
    app.hostname = vm["hostname"].as<string>();
    app.portno = vm["portno"].as<int>();
    //ddapp.socketfd = CLIENT_init((char*)app.hostname.c_str(), app.portno, debug);
    //if (app.socketfd < 0) exit(0);
  } else {
    app.net = new Net<float>(app.network);
    app.net->CopyTrainedLayersFrom(app.weights);
    Caffe::set_phase(Caffe::TEST);
    if (app.gpu)
      Caffe::set_mode(Caffe::GPU);
    else
      Caffe::set_mode(Caffe::CPU);
  }

  // send req_type
  app.pl.size = 0;
  // hardcoded for AlexNet
  strcpy(app.pl.req_name, app.task.c_str());
  if (app.task == "imc") app.pl.size = 3 * 227 * 227;
  // hardcoded for DeepFace
  else if (app.task == "face")
    app.pl.size = 3 * 152 * 152;
  // hardcoded for Mnist
  else if (app.task == "dig")
    app.pl.size = 1 * 28 * 28;
  else
    LOG(FATAL) << "Unrecognized task.\n";

  // read in images
  // cmt: using map, cant use duplicate names for images
  // change to other structure (vector) if you want to send the same exact
  // filename multiple times
  vector<pair<string, Mat> > imgs;
  std::ifstream file(app.input.c_str());
  std::string img_file;
  app.pl.num = 0;
  while (getline(file, img_file)) {
    LOG(INFO) << "Reading " << img_file;
    Mat img;
    if (app.task == "dig")
      img = imread(img_file, CV_LOAD_IMAGE_GRAYSCALE);
    else
      img = imread(img_file);

    if (img.channels() * img.rows * img.cols != app.pl.size)
      LOG(ERROR) << "Skipping " << img_file
                 << ", resize to correct dimensions.\n";
    else {
      imgs.push_back(make_pair(img_file, img));
      ++app.pl.num;
    }
  }

  if (app.pl.num < 1) LOG(FATAL) << "No images read!";

  vector<pair<string, Mat> >::iterator it;
  // align facial recognition image
  if (app.task == "face" && vm["align"].as<bool>()) {
    for (it = imgs.begin(); it != imgs.end(); ++it) {
      LOG(INFO) << "Aligning: " << it->first << endl;
      preprocess(it->second, vm["flandmark"].as<string>(),
                 vm["haar"].as<string>());
      // comment in save + view aligned image
      // imwrite(it->first+"_a", it->second);
    }
  }

  // prepare data into array
  app.pl.data = (float*)malloc(app.pl.num * app.pl.size * sizeof(float));
  float* preds = (float*)malloc(app.pl.num * sizeof(float));

  int img_count = 0;
  for (it = imgs.begin(); it != imgs.end(); ++it) {
    int pix_count = 0;
    for (int c = 0; c < it->second.channels(); ++c) {
      for (int i = 0; i < it->second.rows; ++i) {
        for (int j = 0; j < it->second.cols; ++j) {
          Vec3b pix = it->second.at<Vec3b>(i, j);
          float* p = (float*)(app.pl.data);
          p[img_count * app.pl.size + pix_count] = pix[c];
          ++pix_count;
        }
      }
    }
    ++img_count;
  }
/*
	//Create log file
   pFile = fopen("out.txt","w");
   if(pFile == NULL){
         LOG(INFO) << "Could not create out file";
         return 0;
         }
*/
  if (app.djinn) {

  for(int i = 0; i < RPS; i++)
  {
	pthread_create(threads+i,NULL,create_thread_socket,NULL);
  }

   for (int i=0; i < RPS; i++)
        pthread_join(threads[i], NULL);

   reset_djinn();
/*
    for (int i=0dd; i<seconds; i++) 
    SOCKET_send(app.socketfd, (char*)&app.pl.req_name, MAX_REQ_SIZE, debug);

    // send len
    SOCKET_txsize(app.socketfd, app.pl.num * app.pl.size);

    // send image(s)
    SOCKET_send(app.socketfd, (char*)app.pl.data,
                app.pl.num * app.pl.size * sizeof(float), debug);

    SOCKET_receive(app.socketfd, (char*)preds, app.pl.num * sizeof(float),
                   debug);

    usleep(microseconds);		
    
    SOCKET_close(app.socketfd, debug);

*/
  } else {
    float loss;
    reshape(app.net, app.pl.num * app.pl.size);

    vector<Blob<float>*> in_blobs = app.net->input_blobs();
    in_blobs[0]->set_cpu_data((float*)app.pl.data);
    vector<Blob<float>*> out_blobs = app.net->ForwardPrefilled(&loss);
    memcpy(preds, out_blobs[0]->cpu_data(), app.pl.num * sizeof(float));
  }

  for (it = imgs.begin(); it != imgs.end(); it++) {
    LOG(INFO) << "Image: " << it->first
              << " class: " << preds[distance(imgs.begin(), it)] << endl;
    ;
  }

  if (!app.djinn) free(app.net);
  free(app.pl.data);
  free(preds);
  free(threads);
  return 0;
}
