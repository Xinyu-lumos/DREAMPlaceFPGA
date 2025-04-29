/**
 * @file   timer_cuda.cpp
 * @author Zhili Xiong (DREAMPlaceFPGA-Timing)
 * @date   Mar 2025
 * @brief  Compute timing forward and backward propagation.
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"


DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int timingForwardCudaLauncher(
        const T *x, const T *y, const int *vertex2pin, const int *tnet2src, const int *tnet2dst,
        const int *pin2node_map, const int *flat_levelized_vertices, const int *flat_levelized_vertices_start,
        const int *flat_vertex2pred, const int *flat_vertex2pred_start, const int *flat_vertex2succ,
        const int *flat_vertex2succ_start, const T *vertex_logic_delays, T *vertex_net_delays, 
        T *route_utilization_map, T *pin_utilization_map, const int *is_high_fanout, 
        T *at_vertices, T *rat_vertices, T *slack_tnets,
        T a0, T a1, T bias, T d_r, T d_p, T tau, T route_thresh, T pin_thresh, const T constraint,
        int num_bins_y, int num_vertices, int num_tnets, int max_level, const int SUPER_SOURCE, const int SUPER_SINK
    );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

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
    int deterministic_flag)
{
    CHECK_EVEN(pos);
    CHECK_FLAT(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(vertex2pin);
    CHECK_CONTIGUOUS(vertex2pin);
    CHECK_FLAT(tnet2src);
    CHECK_CONTIGUOUS(tnet2src);
    CHECK_FLAT(tnet2dst);
    CHECK_CONTIGUOUS(tnet2dst);
    CHECK_FLAT(pin2node_map);
    CHECK_CONTIGUOUS(pin2node_map);
    CHECK_FLAT(flat_levelized_vertices);
    CHECK_CONTIGUOUS(flat_levelized_vertices);
    CHECK_FLAT(flat_levelized_vertices_start);
    CHECK_CONTIGUOUS(flat_levelized_vertices_start);
    CHECK_FLAT(flat_vertex2pred);
    CHECK_CONTIGUOUS(flat_vertex2pred);
    CHECK_FLAT(flat_vertex2pred_start);
    CHECK_CONTIGUOUS(flat_vertex2pred_start);
    CHECK_FLAT(flat_vertex2succ);
    CHECK_CONTIGUOUS(flat_vertex2succ);
    CHECK_FLAT(flat_vertex2succ_start);
    CHECK_CONTIGUOUS(flat_vertex2succ_start);
    CHECK_FLAT(vertex_logic_delays);
    CHECK_CONTIGUOUS(vertex_logic_delays);
    CHECK_FLAT(vertex_net_delays);
    CHECK_CONTIGUOUS(vertex_net_delays);

    int num_pins = pos.numel() / 2;
    int max_level = flat_levelized_vertices_start.numel() - 2;
    at::Tensor at_vertices = at::zeros(num_vertices, pos.options());
    at::Tensor rat_vertices = at::ones(num_vertices, pos.options()) * constraint; // the last element is the super sink vertex
    at::Tensor slack_tnets = at::zeros(num_tnets, pos.options());

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "timingForwardCudaLauncher", [&] {
        timingForwardCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(vertex2pin, int),
            DREAMPLACE_TENSOR_DATA_PTR(tnet2src, int),
            DREAMPLACE_TENSOR_DATA_PTR(tnet2dst, int),
            DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_levelized_vertices, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_levelized_vertices_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2pred, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2pred_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2succ, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_vertex2succ_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(vertex_logic_delays, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(vertex_net_delays, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(route_utilization_map, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(is_high_fanout, int), 
            DREAMPLACE_TENSOR_DATA_PTR(at_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(rat_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(slack_tnets, scalar_t),
            a0, a1, bias, d_r, d_p, tau, route_thresh, pin_thresh,
            constraint, num_bins_y, num_vertices, num_tnets, max_level, SUPER_SOURCE, SUPER_SINK);
    });

    return {at_vertices, rat_vertices, slack_tnets};
}


DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::timing_forward, "Timing forward (CUDA)");
}
