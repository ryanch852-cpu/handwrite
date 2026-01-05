import streamlit as st
import cv2
import numpy as np
import os
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# 設定 TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# --- 參數設定 ---
MIN_HEIGHT = 32
MIN_AREA = 140
SHRINK_PX = 4
STABILITY_DURATION = 1.2
MOVEMENT_THRESHOLD = 80

# --- 1. 載入模型 (全域快取) ---
@st.cache_resource
def load_ai_model():
    if os.path.exists("mnist_cnn.h5"):
        try:
            return load_model("mnist_cnn.h5")
        except:
            return None
    return None

model = load_ai_model()

# --- 2. 定義影像處理核心 (類似原本的 Class) ---
class HandwriteProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_boxes = []
        self.stability_start_time = None
        self.is_captured = False
        self.capture_cooldown = 0
        self.captured_frame = None
        
    # 膚色過濾
    def is_valid_content(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return False
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:,:,0])
        mean_s = np.mean(hsv[:,:,1])
        if mean_s > 60: return False
        if 30 < mean_s <= 60:
            if (mean_h < 25 or mean_h > 155): return False
        return True

    # 穩定度檢查
    def check_stability(self, current_boxes):
        if len(current_boxes) == 0:
            self.stability_start_time = None
            return False, 0.0
        
        if len(self.last_boxes) == 0:
            self.last_boxes = current_boxes
            self.stability_start_time = time.time()
            return False, 0.0

        total_movement = 0
        for curr_box in current_boxes:
            c_x, c_y, c_w, c_h = curr_box["box"]
            min_dist = 99999
            for last_box in self.last_boxes:
                l_x, l_y, l_w, l_h = last_box["box"]
                dist = abs(c_x - l_x) + abs(c_y - l_y)
                if dist < min_dist: min_dist = dist
            
            if min_dist < 30: total_movement += min_dist
            else: total_movement += 20 

        count_diff = abs(len(current_boxes) - len(self.last_boxes))
        total_movement += count_diff * 30 
        self.last_boxes = current_boxes

        if total_movement < MOVEMENT_THRESHOLD:
            if self.stability_start_time is None:
                self.stability_start_time = time.time()
            elapsed = time.time() - self.stability_start_time
            progress = min(elapsed / STABILITY_DURATION, 1.0)
            return (elapsed >= STABILITY_DURATION), progress
        else:
            self.stability_start_time = time.time()
            return False, 0.0

    # 每一個影格都會跑進來這裡處理
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 如果處於「已抓拍凍結」狀態，顯示凍結畫面
        current_time = time.time()
        if self.is_captured:
            if current_time < self.capture_cooldown:
                # 保持顯示同一張圖，並顯示倒數
                display_img = self.captured_frame.copy()
                remaining = int(self.capture_cooldown - current_time) + 1
                cv2.putText(display_img, f"CAPTURED! Reset: {remaining}s", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(display_img, format="bgr24")
            else:
                # 時間到，解鎖
                self.is_captured = False
                self.stability_start_time = None
        
        # --- Live 偵測流程 ---
        display_img = img.copy()
        h_f, w_f = img.shape[:2]
        
        # 1. 繪製藍色 ROI 框 (你最想要的！)
        roi_rect = [10, 10, w_f - 20, h_f - 20]
        cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), 
                      (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (255, 0, 0), 2)
        
        # 2. 影像前處理
        roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 18)
        binary_proc = cv2.dilate(thresh, None, iterations=2)
        
        # 3. 找輪廓
        contours, hierarchy = cv2.findContours(binary_proc, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_boxes = []
        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] == -1:
                    area = cv2.contourArea(cnt)
                    if area > MIN_AREA:
                        x, y, w, h = cv2.boundingRect(cnt)
                        has_hole = hierarchy[0][i][2] != -1
                        valid_boxes.append({
                            "box": (x, y, w, h), 
                            "has_hole": has_hole,
                            "aspect_ratio": w / float(h)
                        })
        
        valid_boxes = sorted(valid_boxes, key=lambda b: b["box"][0])
        
        # 4. 批量預測
        batch_rois = []
        batch_info = []
        raw_boxes_for_stability = [] # 用來算穩定度的
        
        for item in valid_boxes:
            x, y, w, h = item["box"]
            rx, ry = x + roi_rect[0], y + roi_rect[1]
            
            # 邊緣過濾
            if x < 15 or y < 15 or (x+w) > binary_proc.shape[1]-15 or (y+h) > binary_proc.shape[0]-15: continue
            if h < MIN_HEIGHT: continue
            
            # 膚色過濾 (在原圖上切)
            roi_color = display_img[ry:ry+h, rx:rx+w]
            if not self.is_valid_content(roi_color): continue
            
            raw_boxes_for_stability.append(item)
            
            # CNN Padding
            roi_single = binary_proc[y:y+h, x:x+w]
            side = max(w, h)
            padding = int(side * 0.2)
            container_size = side + padding * 2
            container = np.zeros((container_size, container_size), dtype=np.uint8)
            offset_y = (container_size - h) // 2
            offset_x = (container_size - w) // 2
            container[offset_y:offset_y+h, offset_x:offset_x+w] = roi_single
            roi_resized = cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA)
            roi_norm = roi_resized.astype('float32') / 255.0
            roi_ready = roi_norm.reshape(28, 28, 1)
            
            batch_rois.append(roi_ready)
            batch_info.append({
                "coords": (rx, ry, w, h),
                "has_hole": item["has_hole"],
                "aspect": item["aspect_ratio"]
            })
            
        # 5. 執行預測與繪圖
        detected_something = False
        if len(batch_rois) > 0 and self.model is not None:
            detected_something = True
            batch_input = np.stack(batch_rois)
            try:
                predictions = self.model.predict(batch_input, verbose=0)
                
                for i, pred in enumerate(predictions):
                    res_id = np.argmax(pred)
                    confidence = np.max(pred)
                    info = batch_info[i]
                    rx, ry, w, h = info["coords"]
                    has_hole = info["has_hole"]
                    aspect = info["aspect"]
                    
                    # 混合修正
                    if res_id == 1:
                        if aspect > 0.45: res_id = 7
                    elif res_id == 7:
                        if aspect < 0.25: res_id = 1
                    if res_id == 7 and has_hole: res_id = 9
                    if res_id == 9 and not has_hole and confidence < 0.95: res_id = 7
                    if res_id == 0 and aspect < 0.5: res_id = 1
                    
                    # 畫綠框 (內縮)
                    draw_x = rx + SHRINK_PX
                    draw_y = ry + SHRINK_PX
                    draw_w = max(1, w - (SHRINK_PX * 2))
                    draw_h = max(1, h - (SHRINK_PX * 2))
                    
                    cv2.rectangle(display_img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), (0, 255, 0), 2)
                    cv2.putText(display_img, str(res_id), (rx, ry-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except:
                pass # 避免 TensorFlow 執行緒衝突

        # 6. 穩定度與進度條
        is_stable, progress = self.check_stability(raw_boxes_for_stability)
        
        # 畫進度條
        bar_w = int(600 * progress)
        color = (0, 255, 255) if progress < 1.0 else (0, 255, 0)
        # 固定在畫面下方
        cv2.rectangle(display_img, (20, h_f - 40), (20 + bar_w, h_f - 25), color, -1)
        cv2.rectangle(display_img, (20, h_f - 40), (620, h_f - 25), (255, 255, 255), 2)
        
        # 觸發抓拍
        if is_stable and detected_something:
            self.is_captured = True
            self.capture_cooldown = time.time() + 3.0 # 凍結 3 秒
            self.captured_frame = display_img.copy() # 存下這一瞬間的畫面
            cv2.putText(display_img, "CAPTURED!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# --- 3. 介面部分 ---
st.set_page_config(page_title="手寫辨識 (Live版)", page_icon="📹", layout="wide")

st.title("📹 手寫數字辨識 (即時影像版)")
st.caption("現在畫面會即時顯示藍框與辨識結果，手穩住後會自動倒數抓拍！")

if model is None:
    st.error("❌ 找不到 `mnist_cnn.h5`！")
    st.stop()

# 啟動 WebRTC 串流
webrtc_ctx = webrtc_streamer(
    key="handwriting-cnn",
    video_processor_factory=HandwriteProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

st.divider()
st.markdown("**操作說明:**")
st.markdown("1. 點擊 `START` 開啟攝影機。")
st.markdown("2. 藍色框框會自動對準畫面。")
st.markdown("3. **將數字卡片拿穩**，下方進度條會開始跑。")
st.markdown("4. 進度條滿了會顯示 **CAPTURED** 並凍結 3 秒方便你看結果。")