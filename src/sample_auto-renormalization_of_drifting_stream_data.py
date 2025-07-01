## -------------------------------------------------------------------------------------------------
## -- Paper      : Online-adaptive data stream processing via cascaded and reverse adaptation
## -- Conference : IEEE Industrial Cyber Physical Systems (ICPS) 2025
## -- Authors    : Detlef Arend, Andreas Schwung
## -- Development: Detlef Arend
## -- Module     : sample_auto-renormalization_of_drifting_stream_data.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-28  0.0.0     DA       Creation
## -- 2024-11-23  1.0.0     DA       Initial implementation
## -- 2024-12-02  1.0.1     DA       Alignment with MLPro 1.9.4
## -- 2025-06-11  1.1.0     DA       Alignment with MLPro 2.0.2
## -- 2025-06-26  1.1.1     DA       Alignment with MLPro 2.0.2 
## -- 2025-06-27  1.1.2     DA       Corrections in class MovingAverage
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2025-06-27)

This sample demonstrates how to auto-renormalize multivariate drifting stream data. It combines
cascaded adaptation with reverse adaptation to focus the entire processing on the buffered data
of a sliding window. 

You will learn:

1) How to set up stream tasks, stream workflows and stream scenarios in MLPro

2) How to visualize stream scenarios

3) How to apply cascaded and reverse adaptation in own applications

"""

from datetime import datetime

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.math import Element
from mlpro.bf.math.properties import Properties
from mlpro.bf.math.geometry import cprop_crosshair
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.streams import InstDict, InstTypeNew, InstTypeDel, Instance
from mlpro.bf.streams.streams import StreamMLProClusterGenerator
from mlpro.bf.streams.tasks import RingBuffer
from mlpro.oa.streams import OAStreamTask, OAStreamWorkflow, OAStreamScenario
from mlpro.oa.streams.tasks import BoundaryDetector, NormalizerMinMax



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MovingAverage (OAStreamTask, Properties):
    """
    Sample implementation of an online-adaptive stream task that buffers internal data relevant for
    a renormalization whenever a prio normalizer changes it's parameters. Here, the moving average
    of the incoming instances is calculated and stored. 
    """

    C_NAME              = 'Moving average'

    C_PROPERTIES        = [ cprop_crosshair ]   

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name = None, 
                  p_range_max = OAStreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0, 
                  p_duplicate_data : bool = False, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_remove_obs : bool = True,
                  **p_kwargs ):
        
        Properties.__init__( self, p_visualize = p_visualize )
       
        OAStreamTask.__init__( self, 
                               p_name = p_name, 
                               p_range_max = p_range_max, 
                               p_ada = p_ada, 
                               p_buffer_size = p_buffer_size, 
                               p_duplicate_data = p_duplicate_data, 
                               p_visualize = p_visualize, 
                               p_logging = p_logging, 
                               **p_kwargs )
                 
        self._moving_avg     = None
        self._num_inst       = 0
        self._remove_obs     = p_remove_obs
        self.crosshair.color = 'red'


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict ):

        # 0 Intro
        inst_avg_id     = -1
        inst_avg_tstamp = None

        
        # 1 Process all incoming new/obsolete stream instances
        for inst_id, (inst_type, inst) in p_instances.items():

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
        p_instances.clear()


        # 3 Add a new stream instance containing the moving average 
        inst_avg_data       = Element( p_set = feature_set )
        inst_avg_data.set_values( p_values = self._moving_avg.copy() )
        inst_avg            = Instance( p_feature_data = inst_avg_data, p_tstamp = inst_avg_tstamp )
        inst_avg.id         = inst_avg_id

        p_instances[inst_avg.id] = ( InstTypeNew, inst_avg )

        self.crosshair.value = self._moving_avg
 

## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        try:
            self._moving_avg = p_normalizer.renormalize( p_data = self._moving_avg.copy() )
            self.log(Log.C_LOG_TYPE_W, 'Moving avg renormalized')
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None):
        OAStreamTask.init_plot( self, p_figure = p_figure, p_plot_settings = p_plot_settings )
        self.crosshair.init_plot( p_figure = self._figure, 
                                  p_plot_settings = self.get_plot_settings() )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_instances : InstDict = None, **p_kwargs):
        OAStreamTask.update_plot( self, p_instances = p_instances, **p_kwargs )
        self.crosshair.update_plot( p_instances = p_instances, **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh = True):
        OAStreamTask.remove_plot(self, p_refresh)
        self.crosshair.remove_plot( p_refresh)


## -------------------------------------------------------------------------------------------------
    def _finalize_plot_view(self, p_inst_ref):
        ps_old = self.get_plot_settings().copy()
        OAStreamTask._finalize_plot_view(self,p_inst_ref)
        ps_new = self.get_plot_settings()

        if ps_new.view != ps_old.view:
            self.crosshair._plot_initialized = False
            self.crosshair.init_plot( p_figure = self._figure, p_plot_settings = ps_new )
 




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DemoScenario (OAStreamScenario):

    C_NAME = 'Normalized drifting data'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode = Mode.C_MODE_SIM, 
                  p_ada : bool = True, 
                  p_cycle_limit : int = 0, 
                  p_num_features : int = 2,
                  p_num_inst : int = 1000,
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):
        
        self._num_features  = p_num_features
        self._num_inst      = p_num_inst

        super().__init__( p_mode = p_mode, 
                          p_ada = p_ada, 
                          p_cycle_limit = p_cycle_limit, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Prepare a native stream from MLPro
        stream = StreamMLProClusterGenerator( p_num_dim = self._num_features,
                                              p_num_instances = self._num_inst,
                                              p_num_clusters = 1,
                                              p_seed = 13,
                                              p_radii = [200,200],
                                              p_velocities = [5],
                                              p_split_and_merge_of_clusters = False,
                                              p_num_of_clusters_for_split_and_merge = 2,
                                              p_logging = Log.C_LOG_NOTHING )


        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAStreamWorkflow( p_name = 'Input signal',
                                     p_range_max = OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada = p_ada,
                                     p_visualize = p_visualize, 
                                     p_logging = p_logging )

     
        # 2.2 Add a basic sliding window to buffer some data
        task_window = RingBuffer( p_buffer_size = 50, 
                                  p_delay = True,
                                  p_enable_statistics = True,
                                  p_name = 'T1 - Sliding window',
                                  p_visualize = p_visualize,
                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task_window )


        # 2.3 Add a moving average task for raw data behind the sliding window
        task_ma1 = MovingAverage( p_name = 'T2 - Moving average (raw)', 
                                  p_ada = p_ada,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task_ma1, p_pred_tasks = [ task_window ] )


        # 2.4 Add a boundary detector and connect to the ring buffer
        task_bd = BoundaryDetector( p_name = 'T3 - Boundary detector', 
                                    p_ada = p_ada, 
                                    p_visualize = p_visualize,
                                    p_logging = p_logging,
                                    p_boundary_provider = task_window )

        workflow.add_task( p_task = task_bd, p_pred_tasks = [task_window] )


        # 2.5 Add a MinMax-Normalizer and connect to the boundary detector
        task_norm_minmax = NormalizerMinMax( p_name = 'T4 - MinMax normalizer', 
                                             p_ada = p_ada, 
                                             p_duplicate_data = True,
                                             p_visualize = p_visualize, 
                                             p_logging = p_logging,
                                             p_dst_boundaries=[-1,1] )

        task_bd.register_event_handler( p_event_id = BoundaryDetector.C_EVENT_ADAPTED, p_event_handler = task_norm_minmax.adapt_on_event )
        workflow.add_task( p_task = task_norm_minmax, p_pred_tasks = [task_bd] )


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
                                  p_logging = p_logging,
                                  p_centroid_crosshair_labels = False )
        
        workflow.add_task( p_task = task_ma4, p_pred_tasks = [ task_norm_minmax ] )
        task_norm_minmax.register_event_handler( p_event_id = NormalizerMinMax.C_EVENT_ADAPTED, p_event_handler = task_ma4.renormalize_on_event )


        # 3 Return stream and workflow
        return stream, workflow




# 1 Demo setup

# 1.1 Default values
num_features    = 2
num_inst        = 500
logging         = Log.C_LOG_WE
step_rate       = 2
view            = PlotSettings.C_VIEW_ND
view_autoselect = True

# 1.2 Welcome message
print('\n\n-----------------------------------------------------------------------------------------')
print('Publication: "Online-adaptive data stream processing via cascaded and reverse adaptation"')
print('Conference : IEEE Industrial Cyber Physical Systems (ICPS) 2025, Emden, Germany')
print('Authors    : Dipl.-Inform. Detlef Arend, Prof. Dr.-Ing. Andreas Schwung')
print('Affiliation: South Westphalia University of Applied Sciences, Germany')
print('Sample     : Auto-renormalization of drifting stream data')
print('-----------------------------------------------------------------------------------------\n')

# 1.3 Get cycle limit from user
i = input(f'Number of stream instances (press ENTER for {num_inst}): ')

if i != '': num_inst = int(i)

# 1.4 Get visualization from user
visualize = input('Visualization Y/N (press ENTER for Y): ').upper() != 'N'

# 1.5 Get number of features and visualization step rate from user
if visualize:
    i = input('Number of features: (press ENTER for 2) ')
    if i != '': num_features = int(i) 

    i = input(f'Visualization step rate (press ENTER for {step_rate}): ')
    if i != '': step_rate = int(i)

# 1.6 Get log level from user
i = input('Log level: "A"=All, "W"=Warnings only, "N"=Nothing (press ENTER for "W"): ').upper() 
if i == 'A': logging = Log.C_LOG_ALL
elif i == 'N': logging = Log.C_LOG_NOTHING



# 2 Instantiate the stream scenario
myscenario = DemoScenario( p_mode = Mode.C_MODE_REAL,
                           p_cycle_limit = num_inst,
                           p_num_features = num_features,
                           p_num_inst = num_inst,
                           p_visualize = visualize,
                           p_logging = logging )



# 3 Reset and run own stream scenario
myscenario.reset()

if visualize:
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = view,
                                                        p_view_autoselect = view_autoselect,
                                                        p_plot_horizon = 100,
                                                        p_data_horizon = num_inst,
                                                        p_step_rate = step_rate ) )

    input('\n\nPlease arrange all windows and press ENTER to start stream processing...')



# 4 Some final statistics
tp_before = datetime.now()
myscenario.run()
tp_after = datetime.now()
tp_delta = tp_after - tp_before
duraction_sec = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.switch_logging( p_logging = Log.C_LOG_TYPE_W)
myscenario.log(Log.C_LOG_TYPE_W, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(num_inst/duraction_sec,2))

input('Press ENTER to exit...')