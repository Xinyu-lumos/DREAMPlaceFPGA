/**
 * @file   timer.cpp
 * @author Zhili Xiong (DREAMPlaceFPGA-Timing)
 * @date   Mar 2025
 * @brief  Compute timing forward and backward propagation.
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"
// local dependency
#include "timing/src/net_delay_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define fpga net delay function
template <typename T>
DEFINE_LINEAR_NET_DELAY_FUNCTION(T);
template <typename T>
DEFINE_CONGESTION_NET_DELAY_FUNCTION(T);

#define INVALID -1  

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute arrival time for each vertex in the timing graph.
/// @param x x location of pins.
/// @param y y location of pins.
/// @param vertex2pin map vertex to pin.
/// @param pin2node_map map pin to node.
/// @param flat_levelized_vertices flat tensor of levelized vertices.
/// @param flat_levelized_vertices_start start index of each level in flat_levelized_vertices.
/// @param flat_vertex2pred flat tensor of predesessors of each vertex.
/// @param flat_vertex2pred_start flat tensor of successors of each vertex.
/// @param vertex_logic_delays logic delay of each timing vertex at the src.
/// @param vertex_net_delays net delay of each timing vertex at the dst.
/// @param at_vertices arrival times of vertices.
/// @param route_utilization_map route utilization map.
/// @param pin_utilization_map pin utilization map.
/// @param a0 coefficient of x in linear net delay function.
/// @param a1 coefficient of y in linear net delay function.
/// @param bias bias in linear net delay function.
/// @param d_r coefficient of route utilization penalty in congestion net delay.
/// @param d_p coefficient of pin utilization penalty in congestion net delay.
/// @param tau high fanout penalty.
/// @param route_thresh route utilization threshold.
/// @param pin_thresh pin utilization threshold.
/// @param num_bins_y number of sites in y direction.
/// @param SUPER_SOURCE super source vertex id.
/// @param level_id the topology level id of timing graph.
/// @param num_vertices_level number of vertices in the level.
/// @param num_threads number of threads.
/// @return 0 if successfully done.
template <typename T>
int computeArrivalTimebyLevelLauncher(
    const T *x, const T *y,
    const int *vertex2pin,
    const int *pin2node_map,
    const int *flat_levelized_vertices,
    const int *flat_levelized_vertices_start,
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
    const int num_vertices_level,
    const int num_threads
    )
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_vertices_level / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)

    /// update arrival time for each vertex at level_id 
    for (int i=flat_levelized_vertices_start[level_id]; i<flat_levelized_vertices_start[level_id+1]; i++)
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
    return 0;
}

/// @brief Compute required arrival time for each vertex in the timing graph.
/// @param flat_levelized_vertices flat tensor of levelized vertices.
/// @param flat_levelized_vertices_start start index of each level in flat_levelized_vertices.
/// @param flat_vertex2succ flat tensor of successors of each vertex.
/// @param flat_vertex2succ_start flat tensor of successors of each vertex.
/// @param vertex_logic_delays logic delay of each timing vertex at the src.
/// @param vertex_net_delays net delay of each timing vertex at the dst.
/// @param rat_vertices required arrival times of vertices.
/// @param constraint timing constraint.
/// @param SUPER_SINK super sink vertex id.
/// @param level_id the topology level id of timing graph.
/// @param num_vertices_level number of vertices in the level.
/// @param num_threads number of threads.
/// @return 0 if successfully done.
template <typename T>
int computeRequiredTimebyLevelLauncher(
    const int *flat_levelized_vertices,
    const int *flat_levelized_vertices_start,
    const int *flat_vertex2succ,
    const int *flat_vertex2succ_start,
    const T *vertex_logic_delays,
    const T *vertex_net_delays,
    T *rat_vertices,
    const T constraint,
    const int SUPER_SINK,
    const int level_id,
    const int num_vertices_level,
    const int num_threads
    )
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_vertices_level / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    /// update required time for each vertex at level_id
    for (int i=flat_levelized_vertices_start[level_id]; i<flat_levelized_vertices_start[level_id+1]; i++)
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
    return 0;
}

/// @brief Compute slack for each timing net.
/// slack = rat(dst) - at(src) - logic_delay(src) - net_delay(dst)
/// @param tnet2src map timing net to src vertex id.
/// @param tnet2dst map timing net to dst vertex id.
/// @param at_vertices arrival times of vertices.
/// @param rat_vertices required arrival times of vertices.
/// @param vertex_logic_delays logic delay of each timing vertex at the src.
/// @param vertex_net_delays net delay of each timing vertex at the dst.
/// @param slack_tnets slack of each timing net.
/// @param num_tnets number of timing nets.
/// @param num_threads number of threads.
/// @return 0 if successfully done.
template <typename T>
int computeSlackLauncher(
    const int *tnet2src,
    const int *tnet2dst,
    const T *at_vertices,
    const T *rat_vertices,
    const T *vertex_logic_delays,
    const T *vertex_net_delays,
    T *slack_tnets,
    const int num_tnets,
    const int num_threads
    )
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_tnets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i=0; i<num_tnets; i++)
    {   
        int src_vertex_id = tnet2src[i];
        int dst_vertex_id = tnet2dst[i];

        if (src_vertex_id == INVALID || dst_vertex_id == INVALID)
        {
            continue;
        }

        T slack = rat_vertices[dst_vertex_id] - at_vertices[src_vertex_id] - \
            vertex_logic_delays[src_vertex_id] - vertex_net_delays[dst_vertex_id];
        slack_tnets[i] = slack;
    }
    return 0;
}

/// @brief Report WNS and TNS.
/// @param flat_vertex2pred map vertex to its predecessors.
/// @param flat_vertex2pred_start start index of each vertex in flat_vertex2pred.
/// @param at_vertices arrival times of vertices.
/// @param rat_vertices required arrival times of vertices.
/// @param wns_tns WNS and TNS.
/// @param SUPER_SINK super sink vertex id.
/// @return 0 if successfully done.
template <typename T>
int reportWNSLauncher(
    const int *flat_vertex2pred,
    const int *flat_vertex2pred_start,
    const T *at_vertices,
    const T *rat_vertices,
    T *wns_tns,
    const int SUPER_SINK
    )
{   
    // single thread report WNS and TNS
    for (int i=flat_vertex2pred_start[SUPER_SINK]; i<flat_vertex2pred_start[SUPER_SINK+1]; i++)
    {
        int endpoint = flat_vertex2pred[i];
        T zero = 0;
        T slack = rat_vertices[endpoint] - at_vertices[endpoint];
        wns_tns[0] = DREAMPLACE_STD_NAMESPACE::min(wns_tns[0], slack);
        wns_tns[1] += DREAMPLACE_STD_NAMESPACE::min(zero, slack);
    }
    return 0;
}

/// @brief Compute timing forward and backward propagation.
std::vector<at::Tensor> timing_forward(
    at::Tensor pos,
    at::Tensor vertex2pin,
    at::Tensor tnet2src,
    at::Tensor tnet2dst,
    at::Tensor pin2node_map,
    at::Tensor flat_levelized_vertices,
    at::Tensor flat_levelized_vertices_start,
    at::Tensor flat_vertex2pred,
    at::Tensor flat_vertex2pred_start,
    at::Tensor flat_vertex2succ,
    at::Tensor flat_vertex2succ_start,
    at::Tensor vertex_logic_delays,
    at::Tensor vertex_net_delays,
    at::Tensor route_utilization_map,
    at::Tensor pin_utilization_map,
    at::Tensor is_high_fanout,
    double a0,
    double a1,
    double bias,
    double d_r,
    double d_p,
    double tau,
    double route_thresh,
    double pin_thresh,
    double constraint,
    int num_bins_y,
    int num_vertices,
    int num_tnets,
    int SUPER_SOURCE,
    int SUPER_SINK,
    int num_threads,
    int deterministic_flag)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_pins = pos.numel() / 2;
    int max_level = flat_levelized_vertices_start.numel() - 2;
    at::Tensor at_vertices = at::zeros(num_vertices, pos.options());
    at::Tensor rat_vertices = at::ones(num_vertices, pos.options()) * constraint; // the last element is the super sink vertex
    at::Tensor slack_tnets = at::zeros(num_tnets, pos.options());

    // forward propagation compute arrival time
    // skip the super source in level 0 and timing start point in level 1, since they are already set to 0
    for (int level = 2; level < max_level; level++)
    {   
        int num_vertices_level = flat_levelized_vertices_start[level+1].item<int>() - flat_levelized_vertices_start[level].item<int>();
        // std::cout<< "level: " << level << " num_vertices_level: " << num_vertices_level << std::endl;

        DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeArrivalTimebyLevelLauncher", [&] {
            computeArrivalTimebyLevelLauncher<scalar_t>(
                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
                DREAMPLACE_TENSOR_DATA_PTR(vertex2pin, int),
                DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_levelized_vertices, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_levelized_vertices_start, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2pred, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2pred_start, int),
                DREAMPLACE_TENSOR_DATA_PTR(is_high_fanout, int),
                DREAMPLACE_TENSOR_DATA_PTR(vertex_logic_delays, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(vertex_net_delays, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(at_vertices, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(route_utilization_map, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t),
                a0, a1, bias, d_r, d_p, tau, route_thresh, pin_thresh, num_bins_y,
                SUPER_SOURCE, level, num_vertices_level, num_threads);
        });
    }

    // backward propagation compute required time
    // skip the super sink and timing endpoint in the last level, since it's already set to constraint
    for (int reverse_level = max_level - 2; reverse_level >=1 ; reverse_level--)
    {   
        int num_vertices_level = flat_levelized_vertices_start[reverse_level+1].item<int>() - flat_levelized_vertices_start[reverse_level].item<int>();
        // std::cout<< "level: " << reverse_level << " num_vertices_level: " << num_vertices_level << std::endl;
        
        DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeRequiredTimebyLevelLauncher", [&] {
            computeRequiredTimebyLevelLauncher<scalar_t>(
                DREAMPLACE_TENSOR_DATA_PTR(flat_levelized_vertices, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_levelized_vertices_start, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2succ, int),
                DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2succ_start, int),
                DREAMPLACE_TENSOR_DATA_PTR(vertex_logic_delays, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(vertex_net_delays, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(rat_vertices, scalar_t),
                constraint, SUPER_SINK,
                reverse_level, num_vertices_level, num_threads);
        });
    }

    // compute slack for each timing net
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeSlackLauncher", [&] {
        computeSlackLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(tnet2src, int),
            DREAMPLACE_TENSOR_DATA_PTR(tnet2dst, int),
            DREAMPLACE_TENSOR_DATA_PTR(at_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(rat_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(vertex_logic_delays, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(vertex_net_delays, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(slack_tnets, scalar_t),
            num_tnets, num_threads);
    });

    return {at_vertices, rat_vertices, slack_tnets};
}

/// @brief Report WNS and TNS
at::Tensor report_wns_tns(
    at::Tensor flat_vertex2pred,
    at::Tensor flat_vertex2pred_start,
    at::Tensor at_vertices,
    at::Tensor rat_vertices,
    int SUPER_SINK)
{
    CHECK_FLAT(flat_vertex2pred);
    CHECK_CONTIGUOUS(flat_vertex2pred);

    at::Tensor wns_tns = at::zeros(2, at_vertices.options());

    DREAMPLACE_DISPATCH_FLOATING_TYPES(at_vertices, "reportWNSLauncher", [&] {
        reportWNSLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2pred, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2pred_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(at_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(rat_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(wns_tns, scalar_t),
            SUPER_SINK);
    });

    return wns_tns;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::timing_forward, "Timing forward");
    m.def("report_wns_tns", &DREAMPLACE_NAMESPACE::report_wns_tns, "Report WNS and TNS");
}