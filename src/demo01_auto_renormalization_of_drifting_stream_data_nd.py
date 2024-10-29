## -------------------------------------------------------------------------------------------------
## -- Paper      : Online-adaptive data stream processing via cascaded and reverse adaptation
## -- Conference : IEEE Industrial Cyber Physical Systems (ICPS) 2025
## -- Author     : Detlef Arend
## -- Module     : demo01_auto_renormalization_of_drifting_stream_data_nd.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-28  0.0.0     DA       Creation
## -- 2024-10-29  1.0.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-10-29)

This module demonstrates 

You will learn:

1) How to ...

2) How to ...

3) How to ...

"""

from datetime import datetime

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.math.properties import Properties
from mlpro.bf.streams import *
from mlpro.bf.streams.streams import StreamMLProClusterGenerator
from mlpro.bf.streams.tasks import RingBuffer
from mlpro.oa.streams import OATask, OAWorkflow, OAScenario
from mlpro.oa.streams.tasks import BoundaryDetector, Normalizer, NormalizerMinMax
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import Centroid, cprop_centroid



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MovingAverage (OATask, Properties):

    C_NAME              = 'Moving average'

    C_PROPERTIES        = [ cprop_centroid ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name = None, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0, 
                  p_duplicate_data : bool = False, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_remove_obs : bool = True,
                  **p_kwargs ):
        
        Properties.__init__( self, p_visualize = p_visualize )
       
        OATask.__init__( self, 
                         p_name = p_name, 
                         p_range_max = p_range_max, 
                         p_ada = p_ada, 
                         p_buffer_size = p_buffer_size, 
                         p_duplicate_data = p_duplicate_data, 
                         p_visualize = p_visualize, 
                         p_logging = p_logging, 
                         **p_kwargs )
                 
        self._moving_avg = None
        self._num_inst   = 0
        self._remove_obs = p_remove_obs
        self._legend     = True


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst):

        # 0 Intro
        inst_avg_id     = -1
        inst_avg_tstamp = None

        
        # 1 Process all incoming new/obsolete stream instances
        for inst_id, (inst_type, inst) in p_inst.items():

            feature_data = inst.get_feature_data().get_values()

            if inst_type == InstTypeNew:
                if self._moving_avg is None:
                    self._moving_avg = feature_data.copy() 
                else:
                    self._moving_avg = ( self._moving_avg * self._num_inst + feature_data ) / ( self._num_inst + 1 )

                self._num_inst += 1

            elif ( inst_type == InstTypeDel ) and self._remove_obs:
                self._moving_avg = ( self._moving_avg * self._num_inst - feature_data ) / ( self._num_inst - 1 )
                self._num_inst  -= 1

            if inst_id > inst_avg_id:
                inst_avg_id     = inst_id
                inst_avg_tstamp = inst.tstamp
                feature_set     = inst.get_feature_data().get_related_set()

        if inst_avg_id == -1: return

            
        # 2 Clear all incoming stream instances
        p_inst.clear()


        # 3 Add a new stream instance containing the moving average 
        inst_avg_data       = Element( p_set = feature_set )
        inst_avg_data.set_values( p_values = self._moving_avg.copy() )
        inst_avg            = Instance( p_feature_data = inst_avg_data, p_tstamp = inst_avg_tstamp )
        inst_avg.id         = inst_avg_id

        p_inst[inst_avg.id] = ( InstTypeNew, inst_avg )

        self.centroid.value = self._moving_avg
 

## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        try:
            self._moving_avg = p_normalizer.renormalize( p_data = self._moving_avg.copy() )
            self.log(Log.C_LOG_TYPE_W, 'Moving avg renormalized')
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None):
        OATask.init_plot( self, p_figure = p_figure, p_plot_settings = p_plot_settings )
        Properties.init_plot( self, p_figure = p_figure, p_plot_settings = p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst = None, **p_kwargs):
        OATask.update_plot( self, p_inst = p_inst, **p_kwargs )
        Properties.update_plot( self, p_inst = p_inst, **p_kwargs )
        if self._legend:
            try:
                self.centroid.get_plot_settings().axes.get_legend().remove()
                self._legend = False
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh = True):
        OATask.remove_plot(self, p_refresh)
        Properties.remove_plot(self, p_refresh)


## -------------------------------------------------------------------------------------------------
    def _finalize_plot_view(self, p_inst_ref):
        ps_old = self.get_plot_settings().copy()
        OATask._finalize_plot_view(self,p_inst_ref)
        ps_new = self.get_plot_settings()

        if ps_new.view != ps_old.view:
            self.centroid._plot_initialized = False
            Properties.init_plot( self, p_figure = self._figure, p_plot_settings = ps_new )
 




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DemoScenario (OAScenario):

    C_NAME = 'OA normalization'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Prepare a native stream from MLPro
        stream = StreamMLProClusterGenerator( p_num_dim = 2,
                                              p_num_instances = 1000,
                                              p_num_clusters = 1,
                                              p_seed = 13,
                                              p_radii = [200,200],
                                              p_velocities = [5],
                                              p_split_and_merge_of_clusters = True,
                                              p_num_of_clusters_for_split_and_merge = 2,
                                              p_logging = Log.C_LOG_NOTHING )


        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAWorkflow( p_name = 'Input signal',
                               p_range_max = OAWorkflow.C_RANGE_NONE,
                               p_ada = p_ada,
                               p_visualize = p_visualize, 
                               p_logging = p_logging )

     
        # 2.2 Add a basic sliding window to buffer some data
        task_window = RingBuffer( p_buffer_size = 50, 
                                  p_delay = True,
                                  p_enable_statistics = True,
                                  p_name = 'T1 - Sliding window',
                                  p_duplicate_data = True,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task_window )

        # 2.3 Add a boundary detector and connect to the ring buffer
        task_bd = BoundaryDetector( p_name = 'T2 - Boundary detector', 
                                    p_ada = p_ada, 
                                    p_visualize = p_visualize,
                                    p_logging = p_logging )

        task_window.register_event_handler( p_event_id = RingBuffer.C_EVENT_DATA_REMOVED, p_event_handler = task_bd.adapt_on_event )
        workflow.add_task( p_task = task_bd, p_pred_tasks = [task_window] )

        # 2.4 Add a MinMax-Normalizer and connect to the boundary detector
        task_norm_minmax = NormalizerMinMax( p_name = 'T3 - MinMax normalizer', 
                                             p_ada = p_ada, 
                                             p_duplicate_data = True,
                                             p_visualize = p_visualize, 
                                             p_logging = p_logging )

        task_bd.register_event_handler( p_event_id = BoundaryDetector.C_EVENT_ADAPTED, p_event_handler = task_norm_minmax.adapt_on_event )
        workflow.add_task( p_task = task_norm_minmax, p_pred_tasks = [task_bd] )

        # 2.5 Add a moving average task for raw data behind the sliding window
        task_ma1 = MovingAverage( p_name = 'T4 - Moving average (raw)', 
                                  p_ada = p_ada,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task_ma1, p_pred_tasks = [ task_window ] )

        # 2.6 Add a moving average task for raw data behind the minmax-normalizer without reverse adaptation
        task_ma2 = MovingAverage( p_name = 'T5 - Moving average (normalized +)', 
                                  p_ada = p_ada,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging,
                                  p_remove_obs = False )
        
        workflow.add_task( p_task = task_ma2, p_pred_tasks = [ task_norm_minmax ] )

        # 2.7 Add a moving average task for raw data behind the minmax-normalizer without reverse adaptation
        task_ma3 = MovingAverage( p_name = 'T6 - Moving average (normalized +/-)', 
                                  p_ada = p_ada,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task_ma3, p_pred_tasks = [ task_norm_minmax ] )

        # 2.8 Add a moving average task for raw data behind the sliding window
        task_ma4 = MovingAverage( p_name = 'T7 - Moving average (renormalized +/-)', 
                                  p_ada = p_ada,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task_ma4, p_pred_tasks = [ task_norm_minmax ] )
        task_norm_minmax.register_event_handler( p_event_id = NormalizerMinMax.C_EVENT_ADAPTED, p_event_handler = task_ma4.renormalize_on_event )


        # 3 Return stream and workflow
        return stream, workflow




# 1 Configuration of the demo
logging         = Log.C_LOG_WE #ALL
visualize       = True
cycle_limit     = 500
step_rate       = 1
view            = PlotSettings.C_VIEW_ND
view_autoselect = False


# 2 Instantiate the stream scenario
myscenario = DemoScenario( p_mode = Mode.C_MODE_REAL,
                           p_cycle_limit = cycle_limit,
                           p_visualize = visualize,
                           p_logging = logging )


# 3 Reset and run own stream scenario
myscenario.reset()
myscenario.init_plot( p_plot_settings=PlotSettings( p_view = view,
                                                    p_view_autoselect = view_autoselect,
                                                    p_plot_horizon = 100,
                                                    p_data_horizon = 150,
                                                    p_step_rate = step_rate ) )

input('\nPlease arrange all windows and press ENTER to start stream processing...')


# 4 Some final statistics
tp_before = datetime.now()
myscenario.run()
tp_after = datetime.now()
tp_delta = tp_after - tp_before
duraction_sec = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_W, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))

input('Press ENTER to exit...')