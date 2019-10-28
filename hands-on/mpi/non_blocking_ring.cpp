#include "mpi.h"
#include <iostream>
#include<chrono>
#include <vector>
#include <numeric>

int main(int argc, char *argv[]) {

  int numtasks, rank, next, prev, buf[2], count, tag1=1, tag2=2, tag3=3;
  constexpr int nRequests = 4;
  constexpr int NMES = 10;
  MPI_Request reqs[NMES]; // required variable for non-blocking calls
  MPI_Status stats[nRequests]; // required variable for Waitall routine
  char inmsg[2] = {'x','y'};
  char outmsg[2] = {'x','y'};
  MPI_Status Stat;  // required variable for receive routines

  //  std::cout << "MPI_COMM_WORLD : " << MPI_COMM_WORLD << std::endl;
  MPI_Init(&argc,&argv);
  //  std::cout << "MPI_COMM_WORLD : " << MPI_COMM_WORLD << std::endl;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  float timing[numtasks];
  
  // determine left and right neighbors
  prev = rank-1;
  next = rank+1;
  if (rank == 0)  prev = numtasks - 1;
  if (rank == (numtasks - 1))  next = 0;

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i=0; i<NMES; i++) {
    // post non-blocking receives and sends for neighbors 
    MPI_Irecv(&buf[0], 1, MPI_INT, prev, tag1, MPI_COMM_WORLD, &reqs[i]);
    MPI_Irecv(&buf[1], 1, MPI_INT, next, tag2, MPI_COMM_WORLD, &reqs[i]);
    MPI_Isend(&rank,   1, MPI_INT, prev, tag2, MPI_COMM_WORLD, &reqs[i]);
    MPI_Isend(&rank,   1, MPI_INT, next, tag1, MPI_COMM_WORLD, &reqs[i]);
  }

  // wait for all non-blocking operations to complete
  MPI_Waitall(nRequests, reqs, stats);
  for (int i=0; i<NMES; i++)
    std::cout << "Rank " << rank << " Received " << reqs[i] << " "
	      << stats[i].MPI_SOURCE << "  with tag " << stats[i].MPI_TAG << std::endl;

  MPI_Get_count(stats, MPI_CHAR, &count);
  std::cout << "Rank " << rank << " Received " << count << " bytes from rank " << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  float local_timing = std::chrono::duration<float>(t1 - t0).count();
  std::cout << "rank " << rank << " local timing " << local_timing << " s" << std::endl;
  /*
  if (rank == 0) {
    timing[rank] = local_timing;
    for (int i = 1; i < 4; i++)
      MPI_Recv (&timing[i], 1, MPI_FLOAT, i, tag3, MPI_COMM_WORLD, &Stat);
  } else 
    MPI_Ssend(&local_timing, 1, MPI_FLOAT, 0, tag3, MPI_COMM_WORLD);
  */
  // do some work while sends/receives progress in background
  
  float global_sum = 0.;
  MPI_Reduce(&local_timing, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
	     MPI_COMM_WORLD);

  if (rank == 0) {
    /*
    for (int i = 0; i < 4; i++)
      std::cout << "--> rank " << i << " local timing " << timing[i] << " s" << std::endl;
    float sum = 0;
    for (int i=0; i<4; i++) sum += timing[i];
    //    float global_sum = 0.;
    //    MPI_Reduce(timing, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
    //    	       MPI_COMM_WORLD);
    */
    printf("Total sum = %f, avg = %f\n", global_sum,
	   global_sum / (numtasks * 1));
  }

  // continue - do more work
  MPI_Finalize();

}
