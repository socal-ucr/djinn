#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/list.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#define MAX_QUEUE_SIZE 100
#define MAX_INSTANCES 2
struct Request
{
    unsigned long time;
    int socknum;
    char req_name[5];
    int sock_elts;
    int in_elts;
    int out_elts;
    float* in;
    float* out;
    unsigned int queueTime;
    unsigned int reshapeTime;
    unsigned int GPUTime;
    unsigned int endTime;
};
typedef boost::interprocess::allocator<struct Request, boost::interprocess::managed_shared_memory::segment_manager>  ShmemAllocator;
typedef boost::interprocess::list<struct Request, ShmemAllocator> shmemList;
