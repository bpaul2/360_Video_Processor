import cv2
import numpy as np
import sys
import yaml
import traceback
import time
import multiprocessing as mp
import omnicv as ocv
import argparse
from enum import Enum
from dataclasses import dataclass
from mmpose.apis import MMPoseInferencer

MAX_QUEUE_SIZE = 10

class Sig(Enum):
    '''
    Identifiers for different message sent between processes
    '''
    DATA = 3
    INIT = 2
    NEXT = 1
    EXIT = 0
    ERROR = -1

@dataclass
class Message:
    '''
    Message class for data passing between classes. Contains a Sig value to
    indicate the kind of message and a content value for the data passed
    '''
    sig: Sig
    content:any

class SkeletonTracker:
    """
    Interface for intializing and controlling mmpose's inferencer
    class
    """
    def __init__(self, pose2d):
        self.inferencer = MMPoseInferencer(pose2d)

    def get_predictions(self, frame):
        result_generator = self.inferencer(frame)

        detected_kp = []
        detected_kp_confidences = []
        detected_bb = []
        detected_bb_confidences = []
        for result in result_generator:
            people_keypoints = result['predictions'][0]
            for predictions in people_keypoints:
                kp = predictions['keypoints']
                kp_conf = predictions['keypoint_scores']
                detected_kp.append(kp)
                detected_kp_confidences.append(kp_conf)
                bb = predictions['bbox']
                bb_conf = predictions['bbox_score']
                detected_bb.append(bb)
                detected_bb_confidences.append(bb_conf)

        return np.array(detected_kp), np.array(detected_kp_confidences) / 10, np.array(detected_bb), np.array(detected_bb_confidences) / 10
    
    def visualize_keypoints(self, frame, keypoints, show_text=True):
        for idx_person, person in enumerate(keypoints):
            for point in person:
                cv2.circle(frame, (int(point[0]), int(point[1])),
                        3, (0, 255, 0), -1)
                
                if show_text:
                    cv2.putText(frame, str(idx_person), 
                        (int(point[0]), int(point[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 1, 2)

    def visualize_people(self, frame, bboxes):
        for person in bboxes:
            for points in person:
                cv2.rectangle(
                    frame,
                    (int(points[0]), int(points[1])),
                    (int(points[2]), int(points[3])),
                    color=(255, 0, 0,),
                    thickness=2)

class ObjectTracker:
    """
    Interface for intializing and controlling different opencv
    object trackers 
    """
    def __init__(self, tracker) -> None:
        OBJECT_TRACKERS = {
            "kcf": cv2.TrackerKCF.create(),
            "csrt": cv2.TrackerCSRT.create()
        }
        assert tracker in OBJECT_TRACKERS.keys(), f'Tracker must belong to one of the following: {OBJECT_TRACKERS.keys()}'
        self.tracker = OBJECT_TRACKERS[tracker]
        self.running = False
        self.last_frame = None
        self.bb = None

    def start_tracker(self, frame, init_bb) -> None:
        self.bb = init_bb
        self.tracker.init(frame, init_bb)
        self.running = True
        self.last_frame = frame

    # def update_bb(self, new_bb) -> None:
    #     if new_bb is None:
    #         self.tracker.init(self.last_frame, self.bb)
    #     else:
    #         self.bb = new_bb
    #         self.tracker.init(self.last_frame, self.bb)

    def update_tracker(self, frame, draw_bb=False) -> float:
        ret, box = self.tracker.update(frame)
        d_yaw = 0
        if ret:
            x, y, w, h = [int(v) for v in box]
            centBB_x = x + w/2
            d_x = centBB_x - (frame.shape[1] / 2)
            d_yaw = -d_x / (frame.shape[1] / 2 / np.pi) * 180 / np.pi
            d_yaw *= int(d_yaw >= 0.5 or d_yaw <= -0.5)
            self.bb = [x, y, w, h]
            if draw_bb:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)   
                cv2.circle(frame, (int(centBB_x), int(y + h/2)), 10, (0,0,255), -1)
                cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 10, (255,0,255), -1)
        self.last_frame = frame
        return d_yaw

class ProgressTimer:
    '''
    Tracks and prints the time passed processing a video frame, the percentage 
    of frames processed and the time left until completion
    '''
    def __init__(self, total_vids: int, total_frames: int, current_vid) -> None:
        self.total_frames = total_frames
        self.total_vids = total_vids
        self.curr_vid = current_vid
        self.curr_frame = None
        self.time_start = None
        self.time_curr = None
        self.running = False

    def start_timer(self) -> None:
        self.curr_frame = 0
        self.time_start = time.time()
        self.running = True

    def track_progress(self) -> None:
        if not self.running:
            self.start_timer()
        self.curr_frame += 1
        progress = float(self.curr_frame) / float(self.total_frames) * 100
        self.time_curr = time.time()
        fps = float(self.curr_frame) / (self.time_curr - self.time_start + 0.0001)
        estimate = float(self.total_frames - self.curr_frame) / fps
        hours, seconds = divmod(estimate, 3600)
        minutes, seconds = divmod(seconds, 60)
        print(f'({self.curr_vid} of {self.total_vids}) FPS: {fps:.2f}, Progress: {progress:.2f}%, Estimated {hours:.0f}h {minutes:.0f}m {seconds:.2f}s left', flush=True)

class BaseProcess(mp.Process):
    '''
    Abstract class that extends the multiprocessing process class to handle 
    intialization, looping, and error exceptions
    '''
    def __init__(self, q_comm_out: mp.Queue) -> None:
        self.q_comm_out = q_comm_out    # Communication queue with parent 
                                        # process for messaging Errors
        mp.Process.__init__(self, target=runner, args=(self,))
        
    def run(self) -> None:
        '''
        Launches process with error handling
        '''
        try:
            mp.Process.run(self)
        except Exception as e:
            tb = traceback.format_exc()
            self.q_comm_out.put(Message(Sig.ERROR, [e, tb]))

    def loop(self) -> None:
        '''
        Loops through process main work method until finished
        '''
        f_run = True
        while f_run:
            f_run = self.do_work()

    def do_work(self) -> bool:
        '''
        Abstract method, main process work done per loop. Returns True if
        work is still in progress, False if finished
        '''
        raise NotImplementedError("Invoked Abstract Method")

class WorkerProcess(BaseProcess):
    '''
    Abstract class for establishes message handling and parameter handling 
    to the BaseProcess class
    '''
    def __init__(self, q_comm_out: mp.Queue, q_comm_in: mp.Queue) -> None:
        super().__init__(q_comm_out)
        self.q_comm_in = q_comm_in  # Messages recieved from parent process
        self.init_params = None     # parameters specific to subclass
        self.out_queue_list = []    # List of queues with outgoing messages 

    def do_work(self) -> bool:
        '''
        Receives and handles messages from parent class, performing work
        based on messsage type
        '''
        ret = None
        if not self.q_comm_in.empty():
            m = self.q_comm_in.get()
            if m.sig is Sig.EXIT:
                self.close_out_queues()
                ret = False
            elif m.sig is Sig.INIT:
                self.initialize_params(m.content)
                ret = True
            elif m.sig is Sig.NEXT:
                self.next_content()
                ret = True
            elif m.sig is Sig.ERROR:
                self.q_comm_out.put(m)
                ret = False
            else:
                raise TypeError('Unknown Message Sig Identifier')
            return ret
        else:
            self.main()
            return True
    
    def close_out_queues(self) -> None:
        '''
        Call before stopping program. Consumes remaining items on outgoing queues
        '''
        if self.out_queue_list:
            for q in self.out_queue_list:
                while not q.empty():
                    q.get()

    def parse_message(self, queue: mp.Queue) -> any:
        '''
        Read message from queue and verifies data type
        '''
        m = queue.get()
        if m.sig is Sig.DATA:
            return m.content
        else:
            raise TypeError('Unknown Message Sig Identifier')


    def initialize_params(self, content: any) -> None:
        '''
        Abstract method, intialize or update process parameters
        '''
        raise NotImplementedError("Invoked Abstract Method")
    
    def next_content(self) -> None:
        '''
        Abstract method, handle change of data
        '''
        raise NotImplementedError("Invoked Abstract Method")
    
    def main(self) -> None:
        '''
        Abstract method, perform work specific to subclass
        '''
        raise NotImplementedError("Invoked Abstract Method")
    
class HandlerProcess(BaseProcess):
    '''
    Handles launching, data, and shutdown for worker threads
    '''
    def __init__(self, q_err: mp.Queue, config: any) -> None:
        super().__init__(q_err) 
        self.q_err_parent = q_err   # Queue for passing error messages to parent process
        self.config = config        # Configuration parameters for self and worker processes
        self.config_map_vid = {}    # Maps processes that update per video to their respective config data
        self.config_map_opt = {}    # Maps processes that update once on launch to their respectice config data
        self.p_comms_in = {}        # Incomming process message queues
        self.p_comms_out = {}       # Outgoing process message queues
        self.p_list = None          # List of worker processes
        self.manager = None

    def process_builder(self, key: str, config_info: str, target: object, args: tuple, conditional_run=None, config_vid=False, config_options=False) -> object:
        '''
        Constructs a worker process

        (str) key:                  String identifier
        (str) config_info:          Identifier within config file specifying parameters
        (object) target:            WorkerProcess class
        (tuple) args:               Class initializer arguments
        (bool) conditional_run:     Indicates if process should be built or not
        (bool) config_vid:      Specifies if process parameter are intialized per video

        returns: WorkerProcess class object
        '''
        # if conditional_run is None or conditional_run is True:
        #     self.p_comms_in[key] = self.manager.Queue(MAX_QUEUE_SIZE)
        #     self.p_comms_out[key] = self.manager.Queue(MAX_QUEUE_SIZE)
        #     if config_options:
        #         self.config_map_opt[key] = config_info
        #     else:
        #         self.config_map_vid[key] = config_info
        #     return target(self.p_comms_out[key], self.p_comms_in[key], *args)
        if conditional_run is None or conditional_run is True:
            self.p_comms_in[key] = self.manager.Queue(MAX_QUEUE_SIZE)
            self.p_comms_out[key] = self.manager.Queue(MAX_QUEUE_SIZE)
            if config_options:
                self.config_map_opt[key] = config_info
            if config_vid:
                self.config_map_vid[key] = config_info
            return target(self.p_comms_out[key], self.p_comms_in[key], *args)

    def setup(self) -> None:
        '''
        Create and connect worker processes
        '''
        self.manager = mp.Manager()
        f_show = self.config['options']['show_vid']
        f_export = self.config['options']['export_data']

        queue_ini_frame = self.manager.Queue(MAX_QUEUE_SIZE)
        queue_converted = self.manager.Queue(MAX_QUEUE_SIZE)
        queue_update = self.manager.Queue(MAX_QUEUE_SIZE)
        queue_to_vid = self.manager.Queue(MAX_QUEUE_SIZE) if f_show else None
        queue_export = self.manager.Queue(MAX_QUEUE_SIZE) if f_export else None

        self.p_list = [
            self.process_builder(key='p_read', config_info='input_data', target=ReadProcess, args=(queue_ini_frame,), config_vid=True),
            self.process_builder(key='p_conv', config_info='conversion_info', target=ConversionProcess, args=(queue_converted, queue_ini_frame, queue_update, queue_export), config_vid=True),
            self.process_builder(key='p_detec', config_info='detection_info', target=DetectionProcess, args=(queue_to_vid, queue_converted, queue_update), config_vid=True, config_options=True),
            self.process_builder(key='p_vid', config_info='video_control', target=VideoProcess, args=(queue_to_vid,), conditional_run=f_show, config_options=True),
            self.process_builder(key='p_export', config_info='output_data', target=ExportProcess, args=(queue_export,), conditional_run=f_export, config_vid=True)
        ]

    def poll_comms(self) -> mp.Queue:
        '''
        Poll incoming messages from worker proceses
        '''
        for queue in self.p_comms_out.values():
            if not queue.empty():
                return queue.get()
        return None

    def global_message(self, m: Message) -> None:
        '''
        Send message to all worker processes
        '''
        for queue in self.p_comms_in.values():
            queue.put(m)

    def process_ini(self, vid=None, current_vid=0, options=False) -> None:
        '''
        Intialize or update worker processes
        '''
        if options:
            mapping = self.config_map_opt
            content = self.config['options']
        else:
            assert(not vid is None)
            mapping = self.config_map_vid
            content = vid
            content['conversion_info'].update({
                'total_vids': len(self.config['videos']),
                'total_frames': cv2.VideoCapture(content['input_data']['in_file']).get(cv2.CAP_PROP_FRAME_COUNT),
                'current_vid': current_vid
                })

        for key in self.p_comms_in.keys() & mapping.keys():
            m = Message(Sig.INIT, content[mapping[key]])
            self.p_comms_in[key].put(m)

    def shutdown(self, s: Sig) -> None:
        '''
        Shutdown worker processes
        '''
        while self.p_list:
            self.global_message(Message(Sig.EXIT, None))
            for p in self.p_list:
                if not p is None:
                    if s is Sig.ERROR:
                            p.terminate()
                    elif s is Sig.EXIT:
                            p.join()
                    else:
                        raise TypeError('Unknown Message Sig Identifier')
                self.p_list.remove(p)

    def process_join(self) -> None:
        '''
        Join worker processes
        '''
        for p in self.p_list:
            if not p is None:
                if not p.is_alive():
                    p.join()
                    self.p_list.remove(p)
            else:
                self.p_list.remove(p)

    def do_work(self) -> bool:
        '''
        Launch worker threads and process videos
        '''
        print("Handler: Setup...",flush=True)
        self.setup()

        print(f"Handler: Launching processes...",flush=True)
        for p in self.p_list:
            if not p is None:
                p.start()
        
        num_vids = len(self.config['videos'])
        curr_vid = 0
        self.process_ini(options=True)
        for v in self.config['videos']:
            curr_vid += 1
            self.process_ini(vid=v, current_vid=curr_vid)
            f_skip = False
            while not f_skip:
                m = self.poll_comms()
                if not m is None:
                    if m.sig is Sig.EXIT:
                        self.shutdown(Sig.EXIT)
                        return False
                    elif m.sig is Sig.INIT:
                        raise NotImplementedError
                    elif m.sig is Sig.NEXT:
                        if curr_vid != num_vids:
                            f_skip = True
                    elif m.sig is Sig.ERROR:
                        self.shutdown(m.sig)
                        self.q_err_parent.put(m)
                    else:
                        raise TypeError('Unknown Message Sig Identifier')
        return False

class ReadProcess(WorkerProcess):
    '''
    Reads in 360-degree video frames as equirectangular projections
    '''
    def __init__(self, q_comm_out: mp.Queue, q_comm_in: mp.Queue, q_out: mp.Queue) -> None:
        super().__init__(q_comm_out, q_comm_in)
        self.q_out = q_out
        self.vid_capture = None
        self.curr_vid = 0
    
    def initialize_params(self, content: any) -> None:
        if self.init_params is None:
            self.init_params = {
                'in_file': None
            }
        self.init_params.update(content)
        self.vid_capture = cv2.VideoCapture(self.init_params['in_file'])
        self.curr_vid += 1

    def main(self) -> None:
        if not self.init_params is None:
            ret, frame = self.vid_capture.read()
            if ret:
                if not self.q_out is None:
                    m = Message(Sig.DATA, [self.curr_vid, frame])
                    self.q_out.put(m)
            else:
                self.request_vid()
    
    def request_vid(self) -> None:
        '''
        Request new video from parent
        '''
        print("Requesting new vid",flush=True)
        self.init_params = None
        self.q_comm_out.put(Message(Sig.NEXT, None))

class ConversionProcess(WorkerProcess):
    '''
    Converts equirectangular projections to fisheye and tracks video angle
    '''
    def __init__(self, q_comm_out: mp.Queue, q_comm_in: mp.Queue, q_out: mp.Queue, q_in: mp.Queue, q_update: mp.Queue, q_export: mp.Queue) -> None:
        super().__init__(q_comm_out, q_comm_in)
        self.q_in = q_in
        self.q_out = q_out
        self.q_update = q_update
        self.q_export = q_export
        self.out_queue_list.extend([self.q_in, self.q_update])
        self.current_angle = None
        self.progress_timer = None
        self.first_frame = True
        self.new_content = None
        self.curr_vid = None

    def initialize_params(self, content: any) -> None:
        if self.init_params is None:
            self.init_params = {
                'f': None,
                'a': None,
                'xi': None,
                'initial_angle': None,
                'output_shape': None
            }
            self.curr_vid = 1
            self.update_params(content)
        else:
            self.new_content = content

    def update_params(self, content: any) -> None:
        self.init_params.update(content)
        self.current_angle = self.init_params['initial_angle']
        self.progress_timer = ProgressTimer(
            self.init_params['total_vids'],
            self.init_params['total_frames'],
            self.init_params['current_vid']
            )
        self.progress_timer.start_timer()

    def main(self) -> None:
        if not self.q_in.empty():
            vid_num, frame = self.parse_message(self.q_in)
            if not self.curr_vid is vid_num:
                self.curr_vid = vid_num
                self.update_params(self.new_content)
            angles = self.parse_message(self.q_update) if not self.first_frame else [0,0,0]
            
            new_frame, new_angles = self.convert_frame(frame, angles)
            self.progress_timer.track_progress()

            m_frame = Message(Sig.DATA, [vid_num, new_frame])
            m_angles = Message(Sig.DATA, [vid_num, new_angles])

            if not self.q_out is None:
                self.q_out.put(m_frame)
                if self.first_frame:
                    self.first_frame = False
            if not self.q_export is None:
                self.q_export.put(m_angles)
    
    def convert_frame(self, frame: any, angles: list) -> any:
        '''
        Convert video frame from equirectanglar into fisheye
        '''
        self.current_angle = [x + y for x, y in zip(self.current_angle, angles)]
        new_frame = ocv.fisheyeImgConv().equirect2Fisheye_DS(
            img=frame,
            outShape=self.init_params['output_shape'],
            f=self.init_params['f'],
            a_=self.init_params['a'],
            xi_=self.init_params['xi'],
            angles=self.current_angle
        )
        return new_frame, self.current_angle

class DetectionProcess(WorkerProcess):
    '''
    Performs motion tracking on human targets center to an image
    '''
    def __init__(self, q_comm_out: mp.Queue, q_comm_in: mp.Queue, q_out: mp.Queue, q_in: mp.Queue, q_update: mp.Queue) -> None:
        super().__init__(q_comm_out, q_comm_in)
        self.q_in = q_in
        self.q_out = q_out
        self.q_update = q_update
        self.out_queue_list.extend([self.q_in])
        self.counter = None
        self.curr_vid = None
        self.curr_obj_tracker = None
        self.next_obj_tracker = None
        self.skel_tracker = None

    def initialize_params(self, content: any) -> None:
        if self.init_params is None:
            self.init_params = {
                'tracking_frames_per_detection': 0,
                'tracking_type': 'kcf',
                'bounding_box': [0,0,0,0],
                'ml_detector': None,
                'visualize_bb': True
            }
        self.init_params.update(content)
        self.counter = self.init_params['tracking_frames_per_detection']
        # self.next_obj_tracker = ObjectTracker(self.init_params['tracking_type'])
        if self.skel_tracker is None or 'ml_detector' in content.keys():
            self.skel_tracker = SkeletonTracker(self.init_params['ml_detector'])

    def main(self) -> None:
        if not self.q_in.empty():
            vid_num, frame = self.parse_message(self.q_in)

            # Swap betwen using object tracker and skeleton tracker
            if self.counter <= 0:
                pos_update = self.skeleton_tracking(frame)
                self.counter = self.init_params['tracking_frames_per_detection']
            else:
                pos_update = self.object_tracking(vid_num, frame)
                self.counter -= 1

            m_frame = Message(Sig.DATA, frame)
            m_update = Message(Sig.DATA, pos_update)

            self.q_update.put(m_update)
            if not self.q_out is None:
                self.q_out.put(m_frame)
    
    def object_tracking(self, vid_num: int, frame: any) -> list:
        if vid_num is None or not vid_num is self.curr_vid:
            print(f"Updating Tracker ({self.curr_vid} to {vid_num})", flush=True)
            self.curr_vid = vid_num
            self.curr_obj_tracker = ObjectTracker(self.init_params['tracking_type'])

        if not self.curr_obj_tracker.running:
            print("New Tracker Confirm", flush=True)
            self.curr_obj_tracker.start_tracker(frame,self.init_params['bounding_box'])
        d_yaw = self.curr_obj_tracker.update_tracker(frame, self.init_params['visualize_bb'])

        return 0, 0, float(d_yaw)

    def skeleton_tracking(self, frame: any) -> list:
        det_keys, _, det_bbs, det_conf_bbs = self.skel_tracker.get_predictions(frame)
        if self.init_params['visualize_bb']:
            self.skel_tracker.visualize_keypoints(frame, det_keys, show_text=False)
            self.skel_tracker.visualize_people(frame, det_bbs)

        frame_y, frame_x, _ = frame.shape
        cent_of_img = [frame_y / 2, frame_x / 2]

        min_dist = None
        min_dx = 0
        d_yaw = 0
        for i in range(len(det_bbs)):
            if det_conf_bbs[i] > 0.05:
                bb = det_bbs[i][0]
                if self.init_params['visualize_bb']:
                    cv2.circle(frame, (int((bb[2]-bb[0]) / 2) + int(bb[0]), int((bb[3]-bb[1]) / 2) + int(bb[1])), 10, (0,0,255), -1)
                d_x = cent_of_img[1]-((bb[2]-bb[0]) / 2 + bb[0])
                d_y = cent_of_img[0]-((bb[3]-bb[1]) / 2 + bb[1])
                abs_dist = np.sqrt(d_x**2 + d_y**2)
                if min_dist is None or (abs_dist < min_dist and abs_dist < 150):
                    min_dist = abs_dist
                    min_dx = d_x

        d_yaw = min_dx / (frame_x / 2 / np.pi) * 180 / np.pi

        return 0, 0, float(d_yaw)

class VideoProcess(WorkerProcess):
    '''
    Displays video frames
    '''
    def __init__(self, q_comm_out: mp.Queue, q_comm_in: mp.Queue, q_in: mp.Queue) -> None:
        super().__init__(q_comm_out, q_comm_in)
        self.q_in = q_in
        self.out_queue_list.extend([self.q_in])
    
    def initialize_params(self, content: any) -> None:
        if self.init_params is None:
            self.init_params = {
                'exit key': 'q'
            }
        self.init_params.update(content)
    
    def main(self) -> None:
        if not self.q_in.empty():
            frame = self.parse_message(self.q_in)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord(self.init_params["exit_key"]):
                self.q_comm_out.put(Message(Sig.EXIT, None))

class ExportProcess(WorkerProcess):
    '''
    Exports video angles into a YAML file
    '''
    def __init__(self, q_comm_out: mp.Queue, q_comm_in: mp.Queue, q_in: mp.Queue) -> None:
        super().__init__(q_comm_out, q_comm_in)
        self.q_in = q_in
        self.out_queue_list.extend([self.q_in])
        self.file = None
        self.new_content = None
        self.curr_vid = None

    def initialize_params(self, content: any) -> None:
        if self.init_params is None:
            self.init_params = {
                'out_file': None
            }
            self.curr_vid = 1
            self.update_params(content)
        else:
            self.new_content = content
    
    def update_params(self, content: any) -> None:
        self.init_params.update(content)
        if self.file is not None:
            self.file.close()
        self.file = open(self.init_params['out_file'], 'w')
    
    def main(self) -> None:
        if not self.q_in.empty():
            vid_num, data = self.parse_message(self.q_in)
            if not self.curr_vid is vid_num:
                self.curr_vid = vid_num
                self.update_params(self.new_content)
            self.export_data(data)
    
    def export_data(self, data: any) -> None:
        yaml.safe_dump(data=data, stream=self.file, explicit_start=True, default_flow_style=False)

def runner(p: BaseProcess) -> None:
    """
    Picklable function that launches the main loop for a process
    """
    p.loop()

if __name__ == "__main__":
    file_default = 'config.yml'
    parser = argparse.ArgumentParser(
                prog='360_video_processor',
                description='Processes 360 videos, tracking the center figure and exporting the camera angles'
                )
    parser.add_argument('-o', '--options', default=file_default)
    args = parser.parse_args()

    try:
        config = yaml.safe_load(open(args.options))
    except FileNotFoundError as e:
        e.add_note("Could not read config file. Please specify a valid config file via '-o' or '--options'")
        raise e

    print("-- Starting Program --",flush=True)
    m = mp.Manager()
    q_err = m.Queue(MAX_QUEUE_SIZE)
    p_handler = HandlerProcess(q_err=q_err, config=config)
    p_handler.start()

    try:
        while True:
            if not q_err.empty():
                m = q_err.get()
                if m.sig is Sig.ERROR:
                    e, tb = m.content
                    p_handler.terminate()
                    print(tb, flush=True)
                    raise Exception('Error from process') from e
                else:
                    raise TypeError('Unknown Message Sig Identifier')
            else:
                if not p_handler.is_alive():
                    p_handler.join()
                    print("-- Program Done --", flush=True)
                    break

    except KeyboardInterrupt:
        p_handler.join()
        print("-- Processes terminated from interrupt --", flush=True)

    sys.exit()