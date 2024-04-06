from flask import Flask, jsonify, render_template, Response, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
from datetime import datetime
import os
import time
import json
import numpy as np
from mmdeploy_runtime import PoseTracker
import subprocess
from VoiceGraph import AudioProcessor
from fight_detect_class_v3 import PersonKeypoints

class Fight_Detect:
    def __init__(self):
        self.fddm = PersonKeypoints()
        
        self.app = Flask(__name__, template_folder="D:/rtm/mmpose/mmdeploy/demo/python/Suviliance-sys/templates")
        self.socketio = SocketIO(self.app)
        self.app.add_url_rule('/start_recording/<int:camera_id>', 'start_recording', self.start_recording, methods=['GET'])
        self.app.add_url_rule('/stop_recording/<int:camera_id>', 'stop_recording', self.stop_recording, methods=['GET'])
        self.app.add_url_rule('/video/<int:camera_id>', 'video', self.video)
        self.app.add_url_rule('/', 'index', self.index)
        self.socketio.on('your_event', self.handle_my_custom_event)
        self.camera_records = {}
        self.FRAME_RATE = 20.0
        self.RESOLUTION = (640, 480)
        self.device_name = 'cuda'
        self.det_model = r'D:/rtm/mmpose/mmdeploy/rtmpose-trt/rtmdet-m'
        self.pose_model = r'D:/rtm/mmpose/mmdeploy/rtmpose-trt/rtmpose-m'
        self.skeleton = 'halpe26'
        self.json = r'D:/rtm/1.json'
        
        self.np = np
        self.np.set_printoptions(precision=4, suppress=True)
        self.VISUALIZATION_CFG = dict(
                halpe26=dict(
                    skeleton=[(15, 13), (13, 11), (11,19),(16, 14), (14, 12), (12,19),
                                (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                                (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                                (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
                    palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0)],
                    link_color=[
                        1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
                    ],
                    point_color=[
                        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
                    ],
                    sigmas=[
                        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
                        0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
                    ]))
    
    def visualize(self, frame, results, frame_id, kp, fps,json, thr=0.5, skeleton_type='halpe26'):
        skeleton = self.VISUALIZATION_CFG[skeleton_type]['skeleton']
        palette = self.VISUALIZATION_CFG[skeleton_type]['palette']
        link_color = self.VISUALIZATION_CFG[skeleton_type]['link_color']
        point_color = self.VISUALIZATION_CFG[skeleton_type]['point_color']

        keypoints, bboxes, _ = results
        scores = keypoints[..., 2]
        keypoints = keypoints[..., :2].astype(int)
        
        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            show = [1] * len(kpts)
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > thr and score[v] > thr:
                    cv2.line(frame, kpts[u], tuple(kpts[v]), palette[color], 1, cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0
            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(frame, kpt, 1, palette[color], 2, cv2.LINE_AA)
        cv2.putText(frame, 'people: '+ str(len(keypoints)), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (100,100,100), 1)
        
        kp.append([{'frame_index': frame_id, 'keypoints': keypoints.tolist(), 'keypoint_scores':scores.tolist()}])
        with open(self.json, 'w') as j:
            json.dump(kp, j, indent = 2)
        
        if cv2.waitKey(1) == ord('q'):
            return (kp, False)
        else:
            return (kp, True)
    @self.socketio.on('your_event')
    def handle_my_custom_event(json):
        print('received json: ' + str(json))
        emit('response', {'response': 'my response'})
    def generate_frames(self, camera_id):
        if camera_id not in self.camera_records:
            self.camera_records[camera_id] = {'is_recording': False, 'camera': cv2.VideoCapture(camera_id)}
        camera = self.camera_records[camera_id]['camera']
        kp = []
        tracker = PoseTracker(det_model=self.det_model, pose_model=self.pose_model, device_name=self.device_name)
        frame_id = 0
        sigmas = self.VISUALIZATION_CFG[self.skeleton]['sigmas']
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
        start_time = time.time()
        num_frames = 0
        fps = 30
        while True:
            success, frame = camera.read()
            if not success:
                break
            results = tracker(state, frame, detect=-1)
            kp, bl = self.visualize(frame, results, frame_id, kp, fps,json, skeleton_type=self.skeleton)
            result_lstm = lstm_model_predict.update_keypoints(kp)
            if type(result_lstm) == bool and result_lstm:
                notifier.send_notification("Violent alert!!!")
            elif type(result_lstm) == str:
                notifier.send_notification(result_lstm)
            num_frames += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = num_frames / elapsed_time
                start_time = time.time()
                num_frames = 0
            if not bl:
                break
            frame_id += 1
            
            if self.camera_records[camera_id]['is_recording']:
                video_writer = self.camera_records[camera_id]['video_writer']
                video_writer.write(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def send_notification(self, message):
        self.socketio.emit('notification', {'message': message}) # 發送網頁通知
    
    def start_video_writer(self, camera_id):
        output_dir_base = os.path.join(self.app.root_path, 'static', 'videos')
        output_dir = os.path.join(output_dir_base, f"camera_{camera_id}")  # 每个相機的專用文件夾
        os.makedirs(output_dir, exist_ok=True)  # 如果目錄不存在，則創建它
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        filepath = os.path.join(output_dir, filename)
        command = [
            'ffmpeg',
            '-y',  # 覆盖輸出文件（如果存在）
            '-f', 'rawvideo',  # 輸入格式
            '-vcodec','rawvideo',  # 輸入编解碼器
            '-pix_fmt', 'bgr24',  # OpenCV的默認像素格式
            '-s', '{}x{}'.format(*self.RESOLUTION),  # 分辨率
            '-r', str(self.FRAME_RATE),  # 幀率
            '-i', '-',  # 從stdin讀取輸入
            '-c:v', 'libx264',  # 輸出視频编解碼器
            '-pix_fmt', 'yuv420p',  # 輸出像素格式
            '-preset', 'ultrafast',  # 編碼速度與壓縮率的平衡
            '-f', 'mp4',  # 輸出格式
            filepath
        ]
        p = subprocess.Popen(command, stdin=subprocess.PIPE)
        return p, filepath

    def detect_cameras(self, limit=10):
        available_cameras = []
        #for i in range(limit):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(0)
            cap.release()
        return available_cameras
    def start_recording(self, camera_id):
        camera_id = int(camera_id)
        if camera_id in self.camera_records and not self.camera_records[camera_id].get('is_recording', False):
            video_writer, filename = self.start_video_writer(camera_id)
            self.camera_records[camera_id].update({
                'is_recording': True,
                'filename': filename,
                'video_writer': video_writer,
            })
            return jsonify(success=True, filename=filename)
        elif camera_id not in self.camera_records:
            return jsonify(success=False, message="Camera not initialized for streaming")
        else:
            return jsonify(success=False, message="Recording already started")

    
    def stop_recording(self, camera_id):
        camera_id = int(camera_id)
        if camera_id in self.camera_records and self.camera_records[camera_id]['is_recording']:
            self.camera_records[camera_id]['is_recording'] = False
            # 關閉FFmpeg進程的stdin
            self.camera_records[camera_id]['video_writer'].stdin.close()
            # 等待FFmpe進程结束
            self.camera_records[camera_id]['video_writer'].wait()
            # 釋放相機資源，如果需要
            # camera_records[camera_id]['camera'].release()
            return jsonify(success=True)
        return jsonify(success=False, message="Recording not started or already stopped")
    def list_videos(self, camera_id):
        output_dir_base = os.path.join(self.app.root_path, 'static', 'videos')
        output_dir = os.path.join(output_dir_base, f"camera_{camera_id}")  # 根據相機ID定位文件夹
        if not os.path.exists(output_dir):
            return jsonify(success=False, message="Video directory does not exist")

        videos = [file for file in os.listdir(output_dir)]
        return jsonify(success=True, videos=videos)
    
    def video(self, camera_id):       
        return Response(self.generate_frames(camera_id),
            mimetype='multipart/x-mixed-replace; boundary=frame')

    
    def index(self):
        camera_ids = self.detect_cameras()
        return render_template('page.html', camera_ids=camera_ids)

    def run(self):
        self.socketio.run(self.app, debug=True, threaded=True)

if __name__ == '__main__':
    lstm_model_predict = PersonKeypoints()
    audio_processor = AudioProcessor()
    audio_processor.run()
    cam_stream = Fight_Detect()
    cam_stream.run()