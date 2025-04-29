/**
 * @file   timer_cuda_kernel.cu
 * @author Zhili Xiong (DREAMPlaceFPGA-Timing)
 * @date   Mar 2025
 * @brief  Compute timing forward and backward propagation.
 */

#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
// local dependency
#include "timing/src/net_delay_function.h"


DREAMPLACE_BEGIN_NAMESPACE

/// define fpga net delay function
template <typename T>
inline __device__ DEFINE_LINEAR_NET_DELAY_FUNCTION(T);
template <typename T>
inline __device__ DEFINE_CONGESTION_NET_DELAY_FUNCTION(T);

#define INVALID -1

template <typename T>
__global__ void computeArrivalTimebyLevel(
    const T *x, const T *y,
    const int *vertex2pin,
    const int *pin2node_map,
    const int *flat_levelized_vertices,
    const int *flat_vertex2pred,
    const int *flat_vertex2pred_start,
    const int *is_high_fanout,
    const T *vertex_logic_delays,
    T *vertex_net_delays,
    T *at_vertices,
    T *route_utilization_map,
    T *pin_utilization_map,
    T a0,
    T a1,
    T bias,
    T d_r,
    T d_p,
    T tau,
    T route_thresh,
    T pin_thresh,
    int num_bins_y,
    const int SUPER_SOURCE,
    const int level_id,
    const int num_vertices_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vertices_level)
    {
        int v_vertex_id = flat_levelized_vertices[i];
        int v_pin_id = vertex2pin[v_vertex_id];
        T x_v = x[v_pin_id];
        T y_v = y[v_pin_id];

        /// arrival time from predecessors
        int pred_start = flat_vertex2pred_start[v_vertex_id];
        int pred_end = flat_vertex2pred_start[v_vertex_id+1];
        for (int j=pred_start; j<pred_end; j++)
        {
            int u_vertex_id = flat_vertex2pred[j];
            // skip the super source vertex 
            if (u_vertex_id == SUPER_SOURCE){
                continue;
            }
            
            int u_pin_id = vertex2pin[u_vertex_id];
            T x_u = x[u_pin_id];
            T y_u = y[u_pin_id];
            
            T net_delay = 0;
            T logic_delay = vertex_logic_delays[u_vertex_id];
            // implement the net delay function

            // CASE1: if the edge is a net edge
            // which mean it's src and dst are in different nodes
            if ((pin2node_map[u_pin_id] != pin2node_map[v_pin_id]))
            {
                T net_delay = LINEAR_NET_DELAY(a0, a1, bias, x_u, y_u, x_v, y_v) + \
                CONGESTION_NET_DELAY(x_u, y_u, x_v, y_v, d_r, d_p, route_thresh, pin_thresh, num_bins_y, route_utilization_map, pin_utilization_map);

                // high fanout penalty
                if (is_high_fanout[u_vertex_id] == 1)
                {
                    net_delay *= tau;
                }
                // attach net delay to the sink vertex
                vertex_net_delays[v_vertex_id] = net_delay;
                
                // compute the arrival time
                T v_arrival_time = at_vertices[u_vertex_id] + logic_delay + net_delay;
                at_vertices[v_vertex_id] = v_arrival_time;
            
            // CASE2: if the edge is a gate edge
            } else
            {
                T v_arrival_time = at_vertices[u_vertex_id] + logic_delay;
                at_vertices[v_vertex_id] = DREAMPLACE_STD_NAMESPACE::max(at_vertices[v_vertex_id], v_arrival_time);
            }
        }
    }  
}

template <typename T>
__global__ void computeRequiredTimebyLevel(
    const int *flat_levelized_vertices,
    const int *flat_vertex2succ,
    const int *flat_vertex2succ_start,
    const T *vertex_logic_delays,
    const T *vertex_net_delays,
    T *rat_vertices,
    const T constraint,
    const int SUPER_SINK,
    const int level_id,
    const int num_vertices_level
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_vertices_level)
    {
        int u_vertex_id = flat_levelized_vertices[i];

        /// required time from successors
        int succ_start = flat_vertex2succ_start[u_vertex_id];
        int succ_end = flat_vertex2succ_start[u_vertex_id+1];
        for (int j=succ_start; j<succ_end; j++)
        {
            int v_vertex_id = flat_vertex2succ[j];

            if (v_vertex_id == SUPER_SINK)
            {
                continue;
            }
            
            T remainingRequiredTime = rat_vertices[v_vertex_id] - vertex_net_delays[v_vertex_id] - vertex_logic_delays[u_vertex_id];
            rat_vertices[u_vertex_id] = DREAMPLACE_STD_NAMESPACE::min(rat_vertices[u_vertex_id], remainingRequiredTime);
        }
    }
}

template <typename T>
__global__ void computeSlack(
    const int *tnet2src,
    const int *tnet2dst,
    const T *at_vertices,
    const T *rat_vertices,
    const T *vertex_logic_delays,
    const T *vertex_net_delays,
    T *slack_tnets,
    const int num_tnets
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_tnets)
    {
        int src_vertex_id = tnet2src[i];
        int dst_vertex_id = tnet2dst[i];

        if (src_vertex_id == INVALID || dst_vertex_id == INVALID)
        {
            return;
        }

        T slack = rat_vertices[dst_vertex_id] - at_vertices[src_vertex_id] - \
            vertex_logic_delays[src_vertex_id] - vertex_net_delays[dst_vertex_id];
        slack_tnets[i] = slack;
    }
}


template <typename T>
int timingForwardCudaLauncher(
    const T *x, const T *y, const int *vertex2pin, const int *tnet2src, const int *tnet2dst,
    const int *pin2node_map, const int *flat_levelized_vertices, const int *flat_levelized_vertices_start,
    const int *flat_vertex2pred, const int *flat_vertex2pred_start, const int *flat_vertex2succ,
    const int *flat_vertex2succ_start, const T *vertex_logic_delays, T *vertex_net_delays, 
    T *route_utilization_map, T *pin_utilization_map, const int *is_high_fanout,
    T* at_vertices, T* rat_vertices, T* slack_tnets,
    T a0, T a1, T bias, T d_r, T d_p, T tau, T route_thresh, T pin_thresh, const T constraint,
    int num_bins_y, int num_vertices, int num_tnets, int max_level, const int SUPER_SOURCE, const int SUPER_SINK)
{
    int thread_count = 64;
    int *flat_levelized_vertices_start_h = nullptr;
    cudaMallocHost((void**)&flat_levelized_vertices_start_h, sizeof(int) * (max_level + 2));
    cudaMemcpy(flat_levelized_vertices_start_h, flat_levelized_vertices_start, sizeof(int) * (max_level + 2), cudaMemcpyDeviceToHost);
    // printf("max_level = %d, num_vertices = %d, num_tnets = %d\n", max_level, num_vertices, num_tnets);

    // compute arrival time for each level
    for (int level = 2; level < max_level; level++)
    {
        int num_vertices_level = flat_levelized_vertices_start_h[level + 1] - flat_levelized_vertices_start_h[level];
        int block_count = ceilDiv((num_vertices_level + thread_count - 1), thread_count);
        
        computeArrivalTimebyLevel<<<block_count, thread_count>>>(
            x, y, vertex2pin, pin2node_map,
            flat_levelized_vertices + flat_levelized_vertices_start_h[level],
            flat_vertex2pred, flat_vertex2pred_start,
            is_high_fanout,
            vertex_logic_delays,
            vertex_net_delays,
            at_vertices,
            route_utilization_map,
            pin_utilization_map,
            a0, a1, bias, d_r, d_p, tau, route_thresh, pin_thresh,
            num_bins_y, SUPER_SOURCE, level, num_vertices_level);

        cudaDeviceSynchronize();
        // printf("forward propagation for level %d, num_vertices_level = %d\n", level, num_vertices_level);
    }

    // compute required time for each level
    // skip the super sink and timing endpoint in the last level, since it's already set to constraint
    for (int reverse_level = max_level - 2; reverse_level >=1 ; reverse_level--)
    {
        int num_vertices_level = flat_levelized_vertices_start_h[reverse_level + 1] - flat_levelized_vertices_start_h[reverse_level];
        int block_count = ceilDiv((num_vertices_level + thread_count - 1), thread_count);

        computeRequiredTimebyLevel<<<block_count, thread_count>>>(
            flat_levelized_vertices + flat_levelized_vertices_start_h[reverse_level],
            flat_vertex2succ, flat_vertex2succ_start, 
            vertex_logic_delays, vertex_net_delays, rat_vertices,
            constraint, SUPER_SINK, reverse_level, num_vertices_level);

        cudaDeviceSynchronize();
        // printf("backward propagation for level %d, num_vertices_level = %d\n", reverse_level, num_vertices_level);
    } 

    // compute slack for each timing net
    int block_count = ceilDiv((num_tnets + thread_count - 1), thread_count);
    computeSlack<<<block_count, thread_count>>>(
        tnet2src, tnet2dst,
        at_vertices, rat_vertices,
        vertex_logic_delays, vertex_net_delays,
        slack_tnets, num_tnets);

    // printf("compute slack for %d timing nets\n", num_tnets);
    
    cudaDeviceSynchronize();
    cudaFreeHost(flat_levelized_vertices_start_h);
    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                   \
    template int timingForwardCudaLauncher<T>(                                        \
        const T *x, const T *y,                                                       \
        const int *vertex2pin, const int *tnet2src, const int *tnet2dst,              \
        const int *pin2node_map,                                                      \
        const int *flat_levelized_vertices, const int *flat_levelized_vertices_start, \
        const int *flat_vertex2pred, const int *flat_vertex2pred_start,               \
        const int *flat_vertex2succ, const int *flat_vertex2succ_start,               \
        const T *vertex_logic_delays, T *vertex_net_delays,                           \
        T *route_utilization_map, T *pin_utilization_map, const int *is_high_fanout,  \
        T *at_vertices, T *rat_vertices, T *slack_tnets,                              \
        T a0, T a1, T bias, T d_r, T d_p, T tau, T route_thresh, T pin_thresh,        \
        const T constraint, int num_bins_y, int num_vertices, int num_tnets,          \
        int max_level, const int SUPER_SOURCE, const int SUPER_SINK);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE


