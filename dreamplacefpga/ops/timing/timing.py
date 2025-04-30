##
# @file   timing.py 
# @author Zhili Xiong
# @date   Mar 2023,  updated Mar 2025
# @brief  Main file implementing the net-weighting for timing-driven placement.
#

import os
import math
import sys
import torch
from torch.autograd import Function
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import dreamplacefpga.ops.timing.timer_cpp as timer_cpp
import dreamplacefpga.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplacefpga.ops.timing.timer_cuda as timer_cuda
import pickle
import logging
import time
import pdb

class TimingFeedback(nn.Module):
    def __init__(self, timer, placedb, pin2node_map, tnet2net, tnet_criticality, tnet_weights, criticality_exp, num_threads, device):
        """
        @brief Initialize the feedback module that inherits from the
         base neural network module in torch framework.
        @param timer the Timer python object, incuding the timing model and lookups
        @param tnet2net the mapping from timing net to net
        @param tnet_criticality the criticality of each timing net
        @param tnet_weights the weights of each timing net
        @param criticality_exp the criticality exponent
        @param device the device gpu or cpu
        """
        super(TimingFeedback, self).__init__()
        self.timer = timer
        self.placedb = placedb
        self.pin2node_map = pin2node_map
        self.num_bins_y = placedb.num_routing_grids_y
        self.tnet2net = tnet2net
        self.tnet_criticality = tnet_criticality
        self.tnet_weights = tnet_weights
        self.criticality_exp = criticality_exp
        self.num_tnets=len(self.tnet_weights)
        self.SUPER_SOURCE = self.timer.tgraph.num_vertices
        self.SUPER_SINK = self.timer.tgraph.num_vertices + 1
        self.num_threads = num_threads
        self.device=device

        ### debug ### 
        self.build_tensors()

    def update_timing_old(self, pos, route_utilization_map, pin_utilization_map, route_utilization_thresh_5, pin_utilization_thresh_5):
        self.timer.tgraph.pin_pos = pos.data.clone().cpu().numpy()
        self.timer.tgraph.route_utilization_map = route_utilization_map
        self.timer.tgraph.pin_utilization_map = pin_utilization_map
        self.timer.tgraph.route_utilization_thresh_5 = route_utilization_thresh_5
        self.timer.tgraph.pin_utilization_thresh_5 = pin_utilization_thresh_5

        self.timer.tgraph.reset()

        tt=time.time()
        self.timer.tgraph.compute_arrival_time()
        logging.info("compute arrival time takes %.6f (s)" % (time.time()-tt))

        tt=time.time()
        self.timer.tgraph.compute_required_time()
        logging.info("compute required time takes %.6f (s)" % (time.time()-tt))

        tt=time.time()
        self.timer.tgraph.compute_slack()
        logging.info("compute slack takes %.6f (s)" % (time.time()-tt))
        
        upd_tnet_wts_criticality = apply_net_weighting_old(
            timer=self.timer,
            num_tnets=self.num_tnets,
            tnet2net=self.tnet2net,
            tnet_weights=self.tnet_weights, 
            tnet_criticality=self.tnet_criticality, 
            criticality_exp=self.criticality_exp,
            device=self.device)

        tt=time.time()
        wns, tns = self.timer.tgraph.report_wns_tns()
        logging.info("timing wns %.6f (ps)" % (wns))
        logging.info("timing tns %.6f (ps)" % (tns))

        # report critical path
        tt=time.time()
        self.timer.tgraph.report_critical_path(upd_tnet_wts_criticality[:self.num_tnets], self.device)
        logging.info("report critical path takes %.6f (s)" % (time.time()-tt))

        return upd_tnet_wts_criticality

    def build_tensors(self):
        """
        @brief build timing graph tensors
        @param pos the position of pins
        """
        tt = time.time()
        self.vertex2pin, \
        self.tnet2src, \
        self.tnet2dst, \
        self.flat_vertex2pred, \
        self.flat_vertex2pred_start, \
        self.flat_vertex2succ, \
        self.flat_vertex2succ_start, \
        self.is_high_fanout, \
        self.vertex_logic_delays, \
        self.flat_levelized_vertices, \
        self.flat_levelized_vertices_start = self.timer.tgraph.build_timing_graph_tensors(self.device)
        logging.info("Convert timing graph to tensors takes %.2f seconds" % (time.time() - tt))

    def update_timing(self, pos, route_utilization_map, pin_utilization_map, route_utilization_thresh_5, pin_utilization_thresh_5):
        """
        @brief update timing 
        """
        vertex_net_delays = torch.zeros(self.timer.tgraph.num_vertices, dtype=torch.float32, device=self.device)

        ## =========================== update timing ========================== ##
        tt = time.time()
        if pos.is_cuda:
            at_vertices, rat_vertices, slack_tnets = timer_cuda.forward(
                pos, self.vertex2pin, self.tnet2src, self.tnet2dst, self.pin2node_map, self.flat_levelized_vertices,
                self.flat_levelized_vertices_start, self.flat_vertex2pred, self.flat_vertex2pred_start,
                self.flat_vertex2succ, self.flat_vertex2succ_start, self.vertex_logic_delays, vertex_net_delays,
                route_utilization_map, pin_utilization_map, self.is_high_fanout, float(self.timer.tmodel.a0),
                float(self.timer.tmodel.a1), float(self.timer.tmodel.bias), float(self.timer.tmodel.d_r),
                float(self.timer.tmodel.d_p), self.timer.tgraph.tau, route_utilization_thresh_5, pin_utilization_thresh_5,
                float(self.timer.tgraph.timing_constraint), self.num_bins_y, self.timer.tgraph.num_vertices,
                self.timer.tgraph.num_tnets, self.SUPER_SOURCE, self.SUPER_SINK, 1)
        else:
            at_vertices, rat_vertices, slack_tnets = timer_cpp.forward(
                pos, self.vertex2pin, self.tnet2src, self.tnet2dst, self.pin2node_map, self.flat_levelized_vertices,
                self.flat_levelized_vertices_start, self.flat_vertex2pred, self.flat_vertex2pred_start,
                self.flat_vertex2succ, self.flat_vertex2succ_start, self.vertex_logic_delays, vertex_net_delays,
                route_utilization_map, pin_utilization_map, self.is_high_fanout, float(self.timer.tmodel.a0),
                float(self.timer.tmodel.a1), float(self.timer.tmodel.bias), float(self.timer.tmodel.d_r),
                float(self.timer.tmodel.d_p), self.timer.tgraph.tau, route_utilization_thresh_5, pin_utilization_thresh_5,
                float(self.timer.tgraph.timing_constraint), self.num_bins_y, self.timer.tgraph.num_vertices,
                self.timer.tgraph.num_tnets, self.SUPER_SOURCE, self.SUPER_SINK, self.num_threads, 1)
        
        logging.info("Timing forward takes %.2f seconds" % (time.time() - tt))

        tt = time.time()
        ## ========================= report WNS TNS ========================= ##
        if pos.is_cuda:
            cpu_at_vertices = at_vertices.cpu()
            cpu_rat_vertices = rat_vertices.cpu()

            wns_tns = timer_cpp.report_wns_tns(
                self.flat_vertex2pred.cpu(),
                self.flat_vertex2pred_start.cpu(),
                cpu_at_vertices,
                cpu_rat_vertices,
                self.SUPER_SINK)
        else:
            wns_tns = timer_cpp.report_wns_tns(
                self.flat_vertex2pred,
                self.flat_vertex2pred_start,
                at_vertices,
                rat_vertices,
                self.SUPER_SINK)

        logging.info("Timing wns %.6f (ps)" % (wns_tns[0]))
        logging.info("Timing tns %.6f (ps)" % (wns_tns[1]))
        logging.info("Report wns and tns takes %.2f seconds" % (time.time() - tt))

        ## ======================== update tnet weights ======================= ##
        tt = time.time()
        upd_tnet_wts_criticality = apply_net_weighting(
            timer=self.timer,
            num_tnets=self.num_tnets,
            slack_tnets=slack_tnets,
            wns=wns_tns[0],
            criticality_exp=self.criticality_exp,
            device=self.device)
        logging.info("Update tnet weights takes %.2f seconds" % (time.time() - tt))

        return upd_tnet_wts_criticality
        

    def report_timing_paths(self, path_file, out_file):
        """
        @brief report timing paths
        """
        test_path_delays = []
        with open(path_file, "r") as f:
            paths = f.readlines()
            for path in paths:
                start, end, _ = path.split()
                path_delay = self.timer.tgraph.report_path(start, end)
                test_path_delays.append(path_delay)

        with open(out_file, "w") as f:
            for path_delay in test_path_delays:
                f.write("%s\n" % (path_delay))


def apply_net_weighting_old(timer, num_tnets, tnet2net, tnet_weights, tnet_criticality, criticality_exp, device):
    """
    @brief apply different net_weighting scheme
    @param timer the Timer python object, incuding the timing model and lookups
    @param num_tnets the number of timing nets
    @param tnet2net the mapping from timing net to net
    @param tnet_weights the net weights in placedb
    @prams tnet_criticality the cricality of a net based on slack
    @param criticality_exp the criticality exponent
    @param device the device cpu or gpu
    """

    upd_tnet_wts_criticality = torch.zeros(2*num_tnets, dtype=tnet_weights.dtype, device=device)
    upd_tnet_wts_criticality = vpr_net_weighting_old(timer=timer, num_tnets=num_tnets, tnet2net=tnet2net, tnet_weights=tnet_weights, tnet_criticality=tnet_criticality, criticality_exp=criticality_exp, device=device)
    
    return upd_tnet_wts_criticality

def vpr_net_weighting_old(timer, num_tnets, tnet2net, tnet_weights, tnet_criticality, criticality_exp, device):
    """
    @brief implement the vpr net-weighting, and explore the best timing-driven interval and criticality exponent
    """
    
    Dmax = timer.tgraph.timing_constraint - timer.tgraph.report_wns_tns()[0]
    upd_tnet_wts_criticality = torch.zeros(2*num_tnets, dtype=tnet_weights.dtype, device=device)

    for tnet_id in range(num_tnets):
        if tnet_id in timer.tgraph.tnet2edge:
            slack = timer.tgraph.tnet2edge[tnet_id].slack
        else:
            continue

        # update net_criticality
        upd_tnet_wts_criticality[num_tnets + tnet_id] = 1 - slack/Dmax

        if slack < 0:
            # update net_weights, set the upper bound of wts to 55
            # upd_tnet_wts_criticality[tnet_id] = min(55, pow(upd_tnet_wts_criticality[num_tnets + tnet_id], criticality_exp))
            upd_tnet_wts_criticality[tnet_id] = pow(upd_tnet_wts_criticality[num_tnets + tnet_id], criticality_exp)

    return upd_tnet_wts_criticality


def apply_net_weighting(timer, num_tnets, slack_tnets, wns, criticality_exp, device):
    """
    @brief apply different net_weighting scheme
    @param timer the Timer python object, incuding the timing model and lookups
    @param num_tnets the number of timing nets
    @param tnet2net the mapping from timing net to net
    @param tnet_weights the net weights in placedb
    @prams tnet_criticality the cricality of a net based on slack
    @param criticality_exp the criticality exponent
    @param device the device cpu or gpu
    """

    upd_tnet_wts_criticality = torch.zeros(2*num_tnets, dtype=slack_tnets.dtype, device=device)
    upd_tnet_wts_criticality = vpr_net_weighting(timer=timer, num_tnets=num_tnets, slack_tnets=slack_tnets, wns=wns, criticality_exp=criticality_exp, device=device)
    
    return upd_tnet_wts_criticality

def vpr_net_weighting(timer, num_tnets, slack_tnets, wns, criticality_exp, device):
    """
    @brief implement the vpr net-weighting, and explore the best timing-driven interval and criticality exponent
    """
    
    Dmax = timer.tgraph.timing_constraint - wns
    upd_tnet_wts_criticality = torch.zeros(2*num_tnets, dtype=slack_tnets.dtype, device=device)

    positive_slack_mask = slack_tnets >= 0
    slack_tnets_masked = slack_tnets.masked_fill(positive_slack_mask, 0)
    ## update net_criticality
    upd_tnet_wts_criticality[num_tnets:] = 1 - slack_tnets_masked/Dmax
    upd_tnet_wts_criticality[num_tnets:][positive_slack_mask] = 0
    ## update net_weights
    upd_tnet_wts_criticality[:num_tnets] = torch.pow(upd_tnet_wts_criticality[num_tnets:], criticality_exp)

    return upd_tnet_wts_criticality



