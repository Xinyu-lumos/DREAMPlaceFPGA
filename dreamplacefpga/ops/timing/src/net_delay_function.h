/**
 * @file   net_delay_function.h
 * @author Zhili Xiong (DREAMPlaceFPGA-Timing)
 * @date   Mar 2025
 */

 
 DREAMPLACE_BEGIN_NAMESPACE

 //#define SCALING_OP maxScaling 
#define LINEAR_NET_DELAY linear_net_delay_function
#define CONGESTION_NET_DELAY congestion_net_delay_function
 
 #define DEFINE_LINEAR_NET_DELAY_FUNCTION(T) \
    T linear_net_delay_function(T a0, T a1, T bias, T src_x, T src_y, T dst_x, T dst_y) \
    { \
        T delta_x = DREAMPLACE_STD_NAMESPACE::abs(src_x - dst_x);\
        T delta_y = DREAMPLACE_STD_NAMESPACE::abs(src_y - dst_y);\
        return DREAMPLACE_STD_NAMESPACE::round(a0*delta_x + a1*delta_y) + bias; \
    }

 #define DEFINE_CONGESTION_NET_DELAY_FUNCTION(T) \
    T congestion_net_delay_function(T src_x, T src_y, T dst_x, T dst_y, T d_r, T d_p, \
        T route_thresh, T pin_thresh, int num_bins_y, T *route_utilization_map, T *pin_utilization_map) \
    { \
        int xl = DREAMPLACE_STD_NAMESPACE::min(src_x, dst_x);  \
        int xh = DREAMPLACE_STD_NAMESPACE::max(src_x, dst_x);  \
        int yl = DREAMPLACE_STD_NAMESPACE::min(src_y, dst_y);  \
        int yh = DREAMPLACE_STD_NAMESPACE::max(src_y, dst_y);  \
        int grid_area = (xh-xl+1) * (yh-yl+1);  \
        T sum_route_utilization = 0;  \
        T sum_pin_utilization = 0;  \
        for (int i = xl; i <= xh; i++)  \
        { \
            for (int j = yl; j <= yh; j++) \
            { \
                sum_route_utilization += route_utilization_map[i*num_bins_y+j];  \
                sum_pin_utilization += pin_utilization_map[i*num_bins_y+j];  \
            } \
        } \
        T route_utilization_avg = sum_route_utilization/grid_area;  \
        T pin_utilization_avg = sum_pin_utilization/grid_area;  \
        if (route_utilization_avg > route_thresh && pin_utilization_avg > pin_thresh)  \
        {  \
            return (d_r * route_utilization_avg + d_p * pin_utilization_avg);  \
        } else  \
        {  \
            return 0;  \
        } \
    }
     

 DREAMPLACE_END_NAMESPACE
