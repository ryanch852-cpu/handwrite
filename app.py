import streamlit as st
import cv2
import numpy as np
import os
import time
import av
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# 設定 TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# --- 參數設定 ---
MIN_HEIGHT = 32
MIN_AREA = 140
SHRINK_PX = 4
STABILITY_DURATION = 1.2
MOVEMENT_THRESHOLD = 80

# --- 1. 載入模型 ---
@st.cache_resource
def load_ai_model():
    if os.path.exists("mnist_cnn.h5"):
        try:
            return load_model("mnist_cnn.h5")
        except:
            return None
    return None

model = load_ai_model()

# --- 2. 核心功能: 膚色過濾 ---
def is_valid_content(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    if mean_s > 60: return False
    if 30 < mean_s <= 60:
        if (mean_h < 25 or mean_h > 155): return False
    return True

# --- 3. WebRTC 影像處理器 ---
class HandwriteProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_boxes = []
        self.stability_start_time = None
        self.frozen = False       
        self.frozen_frame = None  
        self.detected_count = 0   

    # 解除凍結
    def resume(self):
        self.frozen = False
        self.stability_start_time = None
        self.last_boxes = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # [狀態 A] 凍結中
        if self.frozen and self.frozen_frame is not None:
            return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
        
        # [狀態 B] Live 偵測
        display_img = img.copy()
        h_f, w_f = img.shape[:2]
        
        # 繪製藍色 ROI 框
        roi_rect = [10, 10, w_f - 20, h_f - 20]
        cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), 
                      (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (255, 0, 0), 2)
        
        # 影像前處理
        roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 18)
        binary_proc = cv2.dilate(thresh, None, iterations=2)
        
        # 找輪廓
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
        
        # 批量預測
        batch_rois = []
        batch_info = []
        raw_boxes_for_stability = [] 
        
        for item in valid_boxes:
            x, y, w, h = item["box"]
            rx, ry = x + roi_rect[0], y + roi_rect[1]
            
            if x < 15 or y < 15 or (x+w) > binary_proc.shape[1]-15 or (y+h) > binary_proc.shape[0]-15: continue
            if h < MIN_HEIGHT: continue
            
            roi_color = display_img[ry:ry+h, rx:rx+w]
            if not is_valid_content(roi_color): continue
            
            raw_boxes_for_stability.append(item)
            
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
            
        detected_count = 0
        detected_something = False
        
        if len(batch_rois) > 0 and self.model is not None:
            detected_something = True
            try:
                batch_input = np.stack(batch_rois)
                predictions = self.model.predict(batch_input, verbose=0)
                
                for i, pred in enumerate(predictions):
                    res_id = np.argmax(pred)
                    confidence = np.max(pred)
                    info = batch_info[i]
                    rx, ry, w, h = info["coords"]
                    has_hole = info["has_hole"]
                    aspect = info["aspect"]
                    
                    if res_id == 1:
                        if aspect > 0.45: res_id = 7
                    elif res_id == 7:
                        if aspect < 0.25: res_id = 1
                    if res_id == 7 and has_hole: res_id = 9
                    if res_id == 9 and not has_hole and confidence < 0.95: res_id = 7
                    if res_id == 0 and aspect < 0.5: res_id = 1
                    
                    draw_x = rx + SHRINK_PX
                    draw_y = ry + SHRINK_PX
                    draw_w = max(1, w - (SHRINK_PX * 2))
                    draw_h = max(1, h - (SHRINK_PX * 2))
                    
                    cv2.rectangle(display_img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), (0, 255, 0), 2)
                    cv2.putText(display_img, str(res_id), (rx, ry-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    detected_count += 1
            except: pass

        self.detected_count = detected_count

        # 穩定度與抓拍
        if len(raw_boxes_for_stability) == 0:
            self.stability_start_time = None
        elif len(self.last_boxes) == 0:
            self.last_boxes = raw_boxes_for_stability
            self.stability_start_time = time.time()
        else:
            total_movement = 0
            for curr_box in raw_boxes_for_stability:
                c_x, c_y, c_w, c_h = curr_box["box"]
                min_dist = 99999
                for last_box in self.last_boxes:
                    l_x, l_y, l_w, l_h = last_box["box"]
                    dist = abs(c_x - l_x) + abs(c_y - l_y)
                    if dist < min_dist: min_dist = dist
                if min_dist < 30: total_movement += min_dist
                else: total_movement += 20 
            
            count_diff = abs(len(raw_boxes_for_stability) - len(self.last_boxes))
            total_movement += count_diff * 30 
            self.last_boxes = raw_boxes_for_stability

            if total_movement < MOVEMENT_THRESHOLD:
                if self.stability_start_time is None: self.stability_start_time = time.time()
                elapsed = time.time() - self.stability_start_time
                progress = min(elapsed / STABILITY_DURATION, 1.0)
                
                bar_w = int(600 * progress)
                color = (0, 255, 255) if progress < 1.0 else (0, 255, 0)
                cv2.rectangle(display_img, (20, h_f - 40), (20 + bar_w, h_f - 25), color, -1)
                cv2.rectangle(display_img, (20, h_f - 40), (620, h_f - 25), (255, 255, 255), 2)
                
                if elapsed >= STABILITY_DURATION and detected_something:
                    self.frozen = True
                    cv2.putText(display_img, "CAPTURED! Waiting for Input...", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    self.frozen_frame = display_img.copy()
            else:
                self.stability_start_time = time.time()

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# --- 4. Streamlit 介面 ---
st.set_page_config(page_title="手寫辨識 (Web 終極版)", page_icon="📝", layout="wide")

if 'stats' not in st.session_state:
    st.session_state['stats'] = {'total': 0, 'correct': 0}
# [新增] 用來重置輸入框的 key
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0

with st.sidebar:
    st.title("🎛️ 控制台")
    app_mode = st.radio("模式選擇", ["📷 攝影機模式 (Live)", "🎨 手寫板模式"])
    st.divider()
    total = st.session_state['stats']['total']
    correct = st.session_state['stats']['correct']
    acc = (correct / total * 100) if total > 0 else 0.0
    st.metric("總數 (Total)", total)
    st.metric("正確 (Correct)", correct)
    st.metric("準確率", f"{acc:.1f}%")
    if st.button("🔄 重置統計"):
        st.session_state['stats'] = {'total': 0, 'correct': 0}
        st.rerun()

st.title("📝 手寫數字辨識系統")

if model is None:
    st.error("❌ 找不到 `mnist_cnn.h5`！")
    st.stop()

# --- 5. 模式分支 ---

if app_mode == "📷 攝影機模式 (Live)":
    st.info("💡 手持紙張保持穩定，進度條滿後會**自動凍結**，輸入成績後按「儲存」解鎖。")
    
    col_spacer1, col_cam, col_spacer2 = st.columns([1, 3, 1])

    with col_cam:
        ctx = webrtc_streamer(
            key="handwrite-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HandwriteProcessor,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

        c1, c2 = st.columns([1, 1])
        
        with c1:
            # 使用 session_state key 來控制重置
            manual_score = st.number_input("✍️ 輸入正確數量", min_value=0, value=0, key=f"score_input_{st.session_state['input_key']}")
        
        with c2:
            st.write("##") 
            if st.button("💾 儲存並繼續 (Save & Resume)", type="primary", use_container_width=True):
                # 1. 先解除凍結 (這行一定要在 rerun 之前！)
                if ctx.video_processor:
                    ctx.video_processor.resume()
                
                # 2. 存成績
                if manual_score > 0:
                    st.session_state['stats']['total'] += manual_score 
                    st.session_state['stats']['correct'] += manual_score
                    st.toast(f"✅ 已記錄 {manual_score} 筆！")
                    time.sleep(0.5)
                    
                    # 更新 key 以重置輸入框
                    st.session_state['input_key'] += 1
                    
                # 3. 最後才重整 (更新側邊欄)
                st.rerun()

elif app_mode == "🎨 手寫板模式":
    st.info("直接在下方書寫，放開滑鼠自動辨識。")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300, width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data.astype(np.uint8)
        if np.max(img_data) > 0:
            if img_data.shape[2] == 4:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            binary_proc = cv2.dilate(gray, None, iterations=1)
            _, binary_proc = cv2.threshold(binary_proc, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            draw_img = img_data.copy()
            detected_count = 0
            batch_rois = []
            batch_coords = []
            
            for cnt in contours:
                if cv2.contourArea(cnt) > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = binary_proc[y:y+h, x:x+w]
                    side = max(w, h)
                    pad = int(side * 0.2)
                    container = np.zeros((side+pad*2, side+pad*2), dtype=np.uint8)
                    ox, oy = (side+pad*2-w)//2, (side+pad*2-h)//2
                    container[oy:oy+h, ox:ox+w] = roi
                    roi_ready = cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA)
                    roi_ready = roi_ready.astype('float32') / 255.0
                    roi_ready = roi_ready.reshape(28, 28, 1)
                    batch_rois.append(roi_ready)
                    batch_coords.append((x, y, w, h))
            
            if len(batch_rois) > 0:
                preds = model.predict(np.stack(batch_rois), verbose=0)
                for i, pred in enumerate(preds):
                    res_id = np.argmax(pred)
                    x, y, w, h = batch_coords[i]
                    asp = w/h
                    if res_id==1 and asp>0.5: res_id=7
                    if res_id==7 and asp<0.3: res_id=1
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(draw_img, str(res_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    detected_count += 1
            
            st.image(draw_img, channels="BGR")
            col1, col2 = st.columns([1,1])
            with col1:
                hw_score = st.number_input("輸入數量", min_value=0, value=detected_count, key="hw_input")
            with col2:
                st.write("##")
                if st.button("儲存成績", key="hw_save"):
                    st.session_state['stats']['total'] += detected_count
                    st.session_state['stats']['correct'] += hw_score
                    st.rerun()