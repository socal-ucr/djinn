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
#include <glog/logging.h>

#include "boost/program_options.hpp"

#include "SENNA_utils.h"
#include "SENNA_Hash.h"
#include "SENNA_Tokenizer.h"
#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"

#include "SENNA_utils.h"
#include "SENNA_nn.h"
#include "socket.h"
#include "tonic.h"

/* fgets max sizes */
#define MAX_TARGET_VB_SIZE 256
#define NUM_CLIENTS 6

using namespace std;
namespace po = boost::program_options;

bool debug;

std::vector<unsigned int> distribution;

TonicSuiteApp app;

int socketfd = -1;
unsigned int total_requests;

  int *chk_labels = NULL;
  int *pt0_labels = NULL;
  int *pos_labels = NULL;
  int *ner_labels = NULL;
  // weights not used
  SENNA_POS *pos;
  SENNA_CHK *chk;
  SENNA_NER *ner;
  SENNA_Tokens *tokens;
  po::variables_map vm;

volatile bool WARMUP = true;
pthread_mutex_t warmup_mutex = PTHREAD_MUTEX_INITIALIZER;
void* sender_thread(void *args)
{
    int input_size;
    int t_id = (intptr_t)args;
    socketfd = CLIENT_init((char*)app.hostname.c_str(), app.portno, 0);

    if(app.task == "pos")
    {
        pos->input_state = SENNA_realloc(
        pos->input_state, sizeof(float),
            (app.pl.num + pos->window_size - 1) *
            (pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size));
        pos->output_state = SENNA_realloc(pos->output_state, sizeof(float),
                                    app.pl.num * pos->output_state_size);

        SENNA_nn_lookup(pos->input_state,
                  pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size,
                  pos->ll_word_weight, pos->ll_word_size, pos->ll_word_max_idx,
                  tokens->word_idx, app.pl.num, pos->ll_word_padding_idx,
                  (pos->window_size - 1) / 2);
        SENNA_nn_lookup(pos->input_state + pos->ll_word_size,
                  pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size,
                  pos->ll_caps_weight, pos->ll_caps_size, pos->ll_caps_max_idx,
                  tokens->caps_idx, app.pl.num, pos->ll_caps_padding_idx,
                  (pos->window_size - 1) / 2);
        SENNA_nn_lookup(pos->input_state + pos->ll_word_size + pos->ll_caps_size,
                  pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size,
                  pos->ll_suff_weight, pos->ll_suff_size, pos->ll_suff_max_idx,
                  tokens->suff_idx, app.pl.num, pos->ll_suff_padding_idx,
                  (pos->window_size - 1) / 2);

        app.pl.data = (char *)malloc(
            app.pl.num * (pos->window_size * (pos->ll_word_size + pos->ll_caps_size +
                                    pos->ll_suff_size)) * sizeof(float));

        for (int idx = 0; idx < app.pl.num; idx++)
        {
            memcpy((char *)(app.pl.data +
                    idx * (pos->window_size) *
                        (pos->ll_word_size + pos->ll_caps_size +
                         pos->ll_suff_size) *
                        sizeof(float)),
           (char *)(pos->input_state +
                    idx * (pos->ll_word_size + pos->ll_caps_size +
                           pos->ll_suff_size)),
           pos->window_size *
               (pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size) *
               sizeof(float));
        }
        //WARMUP
            // send app
            SOCKET_send(socketfd, (char *)&app.pl.req_name, MAX_REQ_SIZE, debug);
            // send len
            SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

            SOCKET_send(socketfd, (char *)app.pl.data,
                 app.pl.num * app.pl.size * sizeof(float), debug);

    }
    else if (app.task =="chk")
    {
        // chk needs internal pos
        TonicSuiteApp pos_app = app;
        pos_app.task = "pos";
        pos_app.network = vm["common"].as<string>() + "configs/" + "pos.prototxt";
        pos_app.weights = vm["common"].as<string>() + "weights/" + "pos.caffemodel";

        pos_app.socketfd = socketfd;
        strcpy(pos_app.pl.req_name, pos_app.task.c_str());
        pos_app.pl.size =
            pos->window_size *
            (pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size);

        SOCKET_send(socketfd, (char *)&pos_app.pl.req_name, MAX_REQ_SIZE,
                  debug);
        // send len
        SOCKET_txsize(socketfd, pos_app.pl.num * pos_app.pl.size);

        pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx,
                                        tokens->suff_idx, pos_app);

        chk->input_state = SENNA_realloc(
        chk->input_state, sizeof(float),
        (app.pl.num + chk->window_size - 1) *
            (chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size));
        chk->output_state = SENNA_realloc(chk->output_state, sizeof(float),
                                    app.pl.num * chk->output_state_size);

        SENNA_nn_lookup(chk->input_state,
                  chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size,
                  chk->ll_word_weight, chk->ll_word_size, chk->ll_word_max_idx,
                  tokens->word_idx, app.pl.num, chk->ll_word_padding_idx,
                  (chk->window_size - 1) / 2);
        SENNA_nn_lookup(chk->input_state + chk->ll_word_size,
                  chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size,
                  chk->ll_caps_weight, chk->ll_caps_size, chk->ll_caps_max_idx,
                  tokens->caps_idx, app.pl.num, chk->ll_caps_padding_idx,
                  (chk->window_size - 1) / 2);
        SENNA_nn_lookup(chk->input_state + chk->ll_word_size + chk->ll_caps_size,
                  chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size,
                  chk->ll_posl_weight, chk->ll_posl_size, chk->ll_posl_max_idx,
                  pos_labels, app.pl.num, chk->ll_posl_padding_idx,
                  (chk->window_size - 1) / 2);

        input_size = chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size;

        app.pl.data = (char *)malloc(app.pl.num * (chk->window_size * input_size) *
                               sizeof(float));

        for (int idx = 0; idx < app.pl.num; idx++)
        {
            memcpy((char *)(app.pl.data +
                    idx * (chk->window_size) * (input_size) * sizeof(float)),
           (char *)(chk->input_state + idx * input_size),
           chk->window_size * input_size * sizeof(float));
        }
        //warmup
            // chk foward pass
            SOCKET_send(socketfd, (char *)&app.pl.req_name, MAX_REQ_SIZE, 0);
            // send len
            SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

            SOCKET_send(socketfd, (char *)app.pl.data,
                    chk->window_size * input_size * sizeof(float) * app.pl.num,
                    debug);
    }
    else
    {
        input_size = ner->ll_word_size + ner->ll_caps_size + ner->ll_gazl_size +
                   ner->ll_gazm_size + ner->ll_gazo_size + ner->ll_gazp_size;

        ner->input_state =
            SENNA_realloc(ner->input_state, sizeof(float),
                    (app.pl.num + ner->window_size - 1) * input_size);
        ner->output_state = SENNA_realloc(ner->output_state, sizeof(float),
                                    app.pl.num * ner->output_state_size);

        SENNA_nn_lookup(ner->input_state, input_size, ner->ll_word_weight,
                  ner->ll_word_size, ner->ll_word_max_idx, tokens->word_idx,
                  app.pl.num, ner->ll_word_padding_idx,
                  (ner->window_size - 1) / 2);
        SENNA_nn_lookup(ner->input_state + ner->ll_word_size, input_size,
                  ner->ll_caps_weight, ner->ll_caps_size, ner->ll_caps_max_idx,
                  tokens->caps_idx, app.pl.num, ner->ll_caps_padding_idx,
                  (ner->window_size - 1) / 2);
        SENNA_nn_lookup(ner->input_state + ner->ll_word_size + ner->ll_caps_size,
                  input_size, ner->ll_gazl_weight, ner->ll_gazl_size,
                  ner->ll_gazl_max_idx, tokens->gazl_idx, app.pl.num,
                  ner->ll_gazt_padding_idx, (ner->window_size - 1) / 2);
        SENNA_nn_lookup(ner->input_state + ner->ll_word_size + ner->ll_caps_size +
                      ner->ll_gazl_size,
                  input_size, ner->ll_gazm_weight, ner->ll_gazm_size,
                  ner->ll_gazm_max_idx, tokens->gazm_idx, app.pl.num,
                  ner->ll_gazt_padding_idx, (ner->window_size - 1) / 2);
        SENNA_nn_lookup(ner->input_state + ner->ll_word_size + ner->ll_caps_size +
                      ner->ll_gazl_size + ner->ll_gazm_size,
                  input_size, ner->ll_gazo_weight, ner->ll_gazo_size,
                  ner->ll_gazo_max_idx, tokens->gazo_idx, app.pl.num,
                  ner->ll_gazt_padding_idx, (ner->window_size - 1) / 2);
        SENNA_nn_lookup(ner->input_state + ner->ll_word_size + ner->ll_caps_size +
                      ner->ll_gazl_size + ner->ll_gazm_size + ner->ll_gazo_size,
                  input_size, ner->ll_gazp_weight, ner->ll_gazp_size,
                  ner->ll_gazp_max_idx, tokens->gazp_idx, app.pl.num,
                  ner->ll_gazt_padding_idx, (ner->window_size - 1) / 2);

        app.pl.data = (char *)malloc(app.pl.num * (ner->window_size * input_size) *
                               sizeof(float));

        for (int idx = 0; idx < app.pl.num; idx++)
        {
            memcpy((char *)(app.pl.data +
                    idx * (ner->window_size) * (input_size) * sizeof(float)),
           (char *)(ner->input_state + idx * input_size),
           ner->window_size * input_size * sizeof(float));
        }
        //Warmup
            // send app
            SOCKET_send(socketfd, (char *)&app.pl.req_name, MAX_REQ_SIZE, debug);
            // send len
            SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

            SOCKET_send(socketfd, app.pl.data,
                app.pl.num * (ner->window_size * input_size * sizeof(float)),
                ner->debug);
    }

    usleep(1000000);
    WARMUP = false;
    printf("Start Thread:%d\n",socketfd);
    struct timespec tStart, tEnd;
    unsigned long time;
    for(unsigned int i=t_id; i < total_requests; i+=NUM_CLIENTS)
    {
        if (app.task == "pos")
        {
            // send app
            SOCKET_send(socketfd, (char *)&app.pl.req_name, MAX_REQ_SIZE, 0);
            // send len
            SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

            SOCKET_send(socketfd, (char *)app.pl.data,
                 app.pl.num * app.pl.size * sizeof(float), 0);
        }
        else if (app.task == "chk")
        {
            // chk foward pass
            SOCKET_send(socketfd, (char *)&app.pl.req_name, MAX_REQ_SIZE, debug);
            // send len
            SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

            SOCKET_send(socketfd, (char *)app.pl.data,
                    chk->window_size * input_size * sizeof(float) * app.pl.num,
                    debug);
        }
        else if (app.task == "ner")
        {
            // send app
            SOCKET_send(socketfd, (char *)&app.pl.req_name, MAX_REQ_SIZE, debug);
            // send len
            SOCKET_txsize(socketfd, app.pl.num * app.pl.size);

            SOCKET_send(socketfd, app.pl.data,
                app.pl.num * (ner->window_size * input_size * sizeof(float)),
                ner->debug);
        }
        for(int j = 0; j+i < total_requests && j < NUM_CLIENTS; j++)
            usleep(distribution[i+j]);

    }

    printf("Close Thread\n");

    return NULL;
}

void * reciever_thread(void *args)
{
    int t_id = (intptr_t)args;

    char* output;
    if(app.task == "pos")
        output = (char*)malloc((pos->output_state_size) * app.pl.num * sizeof(float));
    else if (app.task == "chk")
        output = (char*)malloc((chk->output_state_size) * app.pl.num * sizeof(float));
    else
        output = (char*)malloc((ner->output_state_size) * app.pl.num * sizeof(float));
    while(WARMUP) {};

    printf("RECIEVING\n");

    for(unsigned int i = t_id; i < total_requests+1;i+=NUM_CLIENTS)
    {
        if(app.task == "pos")
        {
            SOCKET_receive(socketfd, (char *)output,
                   app.pl.num * (pos->output_state_size) * sizeof(float),
                   debug);
        }
        else if(app.task == "chk")
        {
            SOCKET_receive(socketfd, (char *)output,
                  chk->output_state_size * sizeof(float) * app.pl.num, debug);
        }
        else
        {
            SOCKET_receive(socketfd, (char *)output,
                   ner->output_state_size * sizeof(float) * app.pl.num,
                   ner->debug);
        }
    }
    printf("RCLOSE:%d\n",t_id);
}

po::variables_map parse_opts(int ac, char **av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "common,c", po::value<string>()->default_value("../../common/"),
      "Directory with configs and weights")(
      "task,t", po::value<string>()->default_value("pos"),
      "Image task: pos (Part of speech tagging), chk (Chunking), ner (Name "
      "Entity Recognition)")("network,n",
                             po::value<string>()->default_value("pos.prototxt"),
                             "Network config file (.prototxt)")(
      "weights,w", po::value<string>()->default_value("pos.caffemodel"),
      "Pretrained weights (.caffemodel)")(
      "input,i", po::value<string>()->default_value("input/small-input.txt"),
      "File with text to analyze (.txt)")(
      "distribution,D", po::value<string>()->default_value("distribution.txt"),
      "File with text to analyze (.txt)")(
      "tid,T", po::value<int>()->default_value("0"),
      "Id of process")

      ("djinn,d", po::value<bool>()->default_value(false),
       "Use DjiNN service?")("hostname,o",
                             po::value<string>()->default_value("localhost"),
                             "Server IP addr")(
          "portno,p", po::value<int>()->default_value(8080), "Server port")

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

int main(int argc, char *argv[]) {
    vm = parse_opts(argc, argv);
    //get distribution
    ifstream inFile;
    inFile.open(vm["distribution"].as<string>().c_str());

    if(!inFile)
    {
        printf("No distribution file\n");
        exit(1);
    }
   

    float x;
    int i = 0;
    while(inFile >> x)
        distribution.push_back((unsigned int)(x*100000.0f));
    total_requests = distribution.size();

  // google::InitGoogleLogging(argv[0]);

  /* SENNA Inits */
  /* options */
  char *opt_path = NULL;
  int opt_usrtokens = 0;

  /* the real thing */
  char target_vb[MAX_TARGET_VB_SIZE];

  /* inputs */
  SENNA_Hash *word_hash = SENNA_Hash_new(opt_path, "hash/words.lst");
  SENNA_Hash *caps_hash = SENNA_Hash_new(opt_path, "hash/caps.lst");
  SENNA_Hash *suff_hash = SENNA_Hash_new(opt_path, "hash/suffix.lst");
  SENNA_Hash *gazt_hash = SENNA_Hash_new(opt_path, "hash/gazetteer.lst");

  SENNA_Hash *gazl_hash = SENNA_Hash_new_with_admissible_keys(
      opt_path, "hash/ner.loc.lst", "data/ner.loc.dat");
  SENNA_Hash *gazm_hash = SENNA_Hash_new_with_admissible_keys(
      opt_path, "hash/ner.msc.lst", "data/ner.msc.dat");
  SENNA_Hash *gazo_hash = SENNA_Hash_new_with_admissible_keys(
      opt_path, "hash/ner.org.lst", "data/ner.org.dat");
  SENNA_Hash *gazp_hash = SENNA_Hash_new_with_admissible_keys(
      opt_path, "hash/ner.per.lst", "data/ner.per.dat");

  /* labels */
  SENNA_Hash *pos_hash = SENNA_Hash_new(opt_path, "hash/pos.lst");
  SENNA_Hash *chk_hash = SENNA_Hash_new(opt_path, "hash/chk.lst");
  SENNA_Hash *ner_hash = SENNA_Hash_new(opt_path, "hash/ner.lst");

  // weights not used
  pos = SENNA_POS_new(opt_path, "data/pos.dat");
  chk = SENNA_CHK_new(opt_path, "data/chk.dat");
  ner = SENNA_NER_new(opt_path, "data/ner.dat");

  /* tokenizer */
  SENNA_Tokenizer *tokenizer =
      SENNA_Tokenizer_new(word_hash, caps_hash, suff_hash, gazt_hash, gazl_hash,
                          gazm_hash, gazo_hash, gazp_hash, opt_usrtokens);

  /* Tonic Suite inits */
  debug = vm["debug"].as<bool>();
  app.task = vm["task"].as<string>();
  app.network =
      vm["common"].as<string>() + "configs/" + vm["network"].as<string>();
  app.weights =
      vm["common"].as<string>() + "weights/" + vm["weights"].as<string>();
  app.input = vm["input"].as<string>();

  // DjiNN service or local?
  app.djinn = vm["djinn"].as<bool>();
  app.gpu = vm["gpu"].as<bool>();

  if (app.djinn) {
    app.hostname = vm["hostname"].as<string>();
    app.portno = vm["portno"].as<int>();
  //  app.socketfd = CLIENT_init(app.hostname.c_str(), app.portno, debug);
  //  if (app.socketfd < 0) exit(0);
  } else {
  //  app.net = new Net<float>(app.network);
    app.net->CopyTrainedLayersFrom(app.weights);
    if (app.gpu)
      Caffe::set_mode(Caffe::GPU);
    else
      Caffe::set_mode(Caffe::CPU);
  }

  strcpy(app.pl.req_name, app.task.c_str());
  if (app.task == "pos")
    app.pl.size = pos->window_size *
                  (pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size);
  else if (app.task == "chk") {
    app.pl.size = chk->window_size *
                  (chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size);
  } else if (app.task == "ner") {
    int input_size = ner->ll_word_size + ner->ll_caps_size + ner->ll_gazl_size +
                     ner->ll_gazm_size + ner->ll_gazo_size + ner->ll_gazp_size;
    app.pl.size = ner->window_size * input_size;
  }

  // read input file
  ifstream file(app.input.c_str());
  string str;
  string text;
  while (getline(file, str)) text += str;

  // tokenize
  tokens = SENNA_Tokenizer_tokenize(tokenizer, text.c_str());
  app.pl.num = tokens->n;

  int tid = vm["tid"].as<int>();
  if (app.pl.num == 0) LOG(FATAL) << app.input << " empty or no tokens found.";

 	if(app.djinn)
  	{
		pthread_t Sender;
        pthread_t Reciever;
	 	pthread_attr_t attr;
 		pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 1024 * 1024);
        pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
        pthread_create(&Sender,&attr,sender_thread,(void*)(intptr_t)tid);
        pthread_create(&Reciever,&attr,reciever_thread,(void*)(intptr_t)tid);
         
        
        pthread_join(Sender, NULL);
        pthread_join(Reciever, NULL);

  	}
  else
  {
      printf("Only Works with remote server\n");
      exit(1);
  }

/*
    for (int i = 0; i < tokens->n; i++) {
    printf("%15s", tokens->words[i]);
    if (app.task == "pos")
      printf("\t%10s", SENNA_Hash_key(pos_hash, pos_labels[i]));
    else if (app.task == "chk")
      printf("\t%10s", SENNA_Hash_key(chk_hash, chk_labels[i]));
    else if (app.task == "ner")
      printf("\t%10s", SENNA_Hash_key(ner_hash, ner_labels[i]));
    printf("\n");
  }
  // end of sentence
  printf("\n");
*/
  // clean up
  SENNA_Tokenizer_free(tokenizer);

  SENNA_POS_free(pos);
  SENNA_CHK_free(chk);
  SENNA_NER_free(ner);

  SENNA_Hash_free(word_hash);
  SENNA_Hash_free(caps_hash);
  SENNA_Hash_free(suff_hash);
  SENNA_Hash_free(gazt_hash);

  SENNA_Hash_free(gazl_hash);
  SENNA_Hash_free(gazm_hash);
  SENNA_Hash_free(gazo_hash);
  SENNA_Hash_free(gazp_hash);

  SENNA_Hash_free(pos_hash);

    SOCKET_close(socketfd, debug);

  if (!app.djinn) free(app.net);

  return 0;
}
