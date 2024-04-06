# v3以每15幀為單位
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class PersonKeypoints:
    def __init__(self):
        self.keypoints_15_list = []
        self.keypoints_5window = []  
        self.person_count = {}
        self.windowsize = 15
        self.overlap = 10
        self.prenumper = 0
        self.count = 0
        self.abnormal_gather = False
        self.model = load_model('LSTM__ds15_nor_1000_best4.h5')
        
    def update_keypoints(self, keypoints):
        if self.count % 2 == 0:
            keypoint_person = keypoints[0]['keypoints']
            self.count += 1
        else:
            self.count += 1
            return False
        
        
        if self.prenumper != len(keypoint_person): # 人數有變動
            # 全部資料刷新
            self.refresh()
            
        self.prenumper = len(keypoint_person) # 紀錄當前人數
        if self.prenumper >= 10:
            return 'Abnormal Crowd Gathering'
        #---------------------------------------------------------------------------------------------
        # 確認有偵測到人
        if self.prenumper >= 1:
            # 初始狀態，尚未儲存任何骨架點
            if len(self.keypoints_15_list) == 0:
                for i in range(self.prenumper):
                    # 確認有抓到26個骨架點，每個骨架點也有確實記錄(x, y)座標
                    if self._is_valid_person(keypoint_person[i]):
                        name = f"person_{i}"
                        
                        self.keypoints_15_list.append([])
                        self.keypoints_15_list[i].append(keypoint_person[i])
                        self.person_count[name] = 1 # 一筆資料
                    # 有點沒抓到
                    else:
                        self.refresh()
            # 不是初始狀態
            else:
                for i in range(self.prenumper):
                    if self._is_valid_person(keypoint_person[i]):
                        name = f"person_{i}"
                        self.keypoints_15_list[i].append(keypoint_person[i])
                        self.person_count[name] += 1
                    # 有點沒抓到
                    else:
                        self.refresh()
        else:
            self.refresh()   
        self.detect() # 檢查有沒有已滿15幀
        
        return self.fight_detect() # 檢查有沒有打架事件發生
            
        #--------------------------------------------------------------------------------------------            
    def _is_valid_person(self, person_keypoints):
        
        return len(person_keypoints) == 26 and all(len(coordinate) == 2 for coordinate in person_keypoints)
    
    
    
    def refresh(self): # 刷新數據
        self.keypoints_15_list = []
        self.person_count = {}
        self.keypoints_5window = []
    
    def detect(self):
        for i in range(len(self.person_count)):
            name = f"person_{i}"
            # 滿15幀
            if len(self.keypoints_15_list[i]) == 15:
                self.keypoints_5window.append([])
                sequence = np.zeros((1, 15, 26, 2))
                for j, outer_list in enumerate(self.keypoints_15_list[i]):
                    for k, inner_list in enumerate(outer_list):
                        sequence[0, j, k, 0] = inner_list[0]
                        sequence[0, j, k, 1] = inner_list[1]
                X_reshaped = sequence.reshape(1, 15, -1)
                num_samples_train, time_steps_train, feature_dim_train = X_reshaped.shape
                X_reshaped_new = X_reshaped.reshape(num_samples_train, -1)
                scaler = MinMaxScaler() # 做歸一化
                X_normalized_flat = scaler.fit_transform(X_reshaped_new)
                X_normalized = X_normalized_flat.reshape(num_samples_train, time_steps_train, feature_dim_train)
                pre = self.model.predict(X_normalized)
                y_pred_binary = (pre > 0.5).astype(int) # 模型預測結果是機率，超過0.5為打架
                self.keypoints_5window[i].append(y_pred_binary[0][0])
                
                self.keypoints_15_list[i] = self.keypoints_15_list[i][-10:] # sliding window
                self.person_count[name] = 10
                
    def fight_detect(self):
        record = False
        for idx, conti_labels in enumerate(self.keypoints_5window):
            if len(conti_labels) == 5: # 5個windows做一次評估
                if sum(conti_labels) >= 3: # 5個windows有三個被判定為打架
                    record = True
                self.keypoints_5window[idx] = self.keypoints_5window[idx][1:5] # 保留四個
        
        return record
                
            
        
            
        