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

# [已改回] 調整藍框大小 (改回原本的 60)
ROI_MARGIN_X = 60   # 左右留白 
ROI_MARGIN_Y = 60   # 上下留白

TEXT_Y_OFFSET = 15 
INFO_PANEL_WIDTH = 450 

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

# --- 2. 核心功能 ---
def is_valid_content(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    if mean_s > 60: return False
    if 30 < mean_s <= 60:
        if (mean_h < 25 or mean_h > 155): return False
    return True

# 手寫模式專用：ID 追蹤與匹配
def update_tracker(contours):
    current_items = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            current_items.append({'cnt': cnt, 'center': (cx, cy), 'id': None})

    used_current_indices = set()
    new_tracker_state = {}
    
    if 'tracker_state' in st.session_state:
        for old_id, old_center in st.session_state['tracker_state'].items():
            min_dist = 9999
            match_idx = -1
            for i, item in enumerate(current_items):
                if i in used_current_indices: continue
                dist = np.hypot(item['center'][0]-old_center[0], item['center'][1]-old_center[1])
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    match_idx = i
            
            if match_idx != -1:
                current_items[match_idx]['id'] = old_id
                used_current_indices.add(match_idx)
                new_tracker_state[old_id] = current_items[match_idx]['center']

    for i, item in enumerate(current_items):
        if item['id'] is None:
            item['id'] = st.session_state['next_id']
            st.session_state['next_id'] += 1
            new_tracker_state[item['id']] = item['center']

    st.session_state['tracker_state'] = new_tracker_state
    current_items.sort(key=lambda x: x['id'])
    return current_items

# --- 3. WebRTC 影像處理器 ---
class HandwriteProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_boxes = []
        self.stability_start_time = None
        self.frozen = False        
        self.frozen_frame = None  
        self.detected_count = 0   
        self.ui_results = [] 

    def resume(self):
        self.frozen = False
        self.stability_start_time = None
        self.last_boxes = []
        self.ui_results = [] 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.frozen and self.frozen_frame is not None:
            return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
        
        display_img = img.copy()
        h_f, w_f = img.shape[:2]
        
        # [修改] 使用 X, Y Margin 來計算 roi_rect
        roi_rect = [ROI_MARGIN_X, ROI_MARGIN_Y, w_f - 2*ROI_MARGIN_X, h_f - 2*ROI_MARGIN_Y]
        
        cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), 
                      (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (255, 0, 0), 2)
        
        roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
        
        if roi_img.size == 0:
             return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 18)
        binary_proc = cv2.dilate(thresh, None, iterations=2)
        
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
        
        batch_rois = []
        batch_info = []
        raw_boxes_for_stability = [] 
        
        for item in valid_boxes:
            x, y, w, h = item["box"]
            rx, ry = x + roi_rect[0], y + roi_rect[1]
            
            if x < 5 or y < 5 or (x+w) > binary_proc.shape[1]-5 or (y+h) > binary_proc.shape[0]-5: continue
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
        current_frame_text_results = []

        if len(batch_rois) > 0 and self.model is not None:
            detected_something = True
            try:
                batch_input = np.stack(batch_rois)
                predictions = self.model.predict(batch_input, verbose=0)
                
                for i, pred in enumerate(predictions):
                    top_indices = pred.argsort()[-3:][::-1]
                    res_id = top_indices[0]
                    confidence = pred[res_id]
                    
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
                    
                    cv2.putText(display_img, f"#{i+1}", (rx, ry-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    info_text = f"**#{i+1}**: 數字 `{res_id}` (信心度: {int(confidence*100)}%)"
                    
                    if confidence < 1.0:
                        alt_id = top_indices[1]
                        alt_conf = pred[alt_id]
                        if alt_conf > 0.01:
                            info_text += f" ⚠️ 其他: `{alt_id}` ({int(alt_conf*100)}%)"
                            
                    current_frame_text_results.append(info_text)
                    detected_count += 1
            except: pass

        self.detected_count = detected_count

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
                
                bar_y = h_f - 20 
                bar_w = int(600 * progress)
                color = (0, 255, 255) if progress < 1.0 else (0, 255, 0)
                
                cv2.rectangle(display_img, (20, bar_y - 15), (20 + bar_w, bar_y), color, -1)
                cv2.rectangle(display_img, (20, bar_y - 15), (w_f - 20, bar_y), (255, 255, 255), 2)
                
                if elapsed >= STABILITY_DURATION and detected_something:
                    self.frozen = True
                    # [修改] 文字位置根據 Y Margin 調整
                    text_y = max(30, ROI_MARGIN_Y - TEXT_Y_OFFSET) 
                    cv2.putText(display_img, "CAPTURED!", (ROI_MARGIN_X, text_y), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    
                    self.frozen_frame = display_img.copy()
                    self.ui_results = current_frame_text_results
            else:
                self.stability_start_time = time.time()

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# --- 4. Streamlit 介面 ---
st.set_page_config(page_title="手寫辨識 (Web 終極版)", page_icon="📝", layout="wide")

if 'stats' not in st.session_state:
    st.session_state['stats'] = {'camera': {'total': 0, 'correct': 0}, 'handwriting': {'total': 0, 'correct': 0}}
if 'history' not in st.session_state:
    st.session_state['history'] = {'camera': [], 'handwriting': []}
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0
if 'canvas_key' not in st.session_state:
    st.session_state['canvas_key'] = "canvas_0"
if 'tracker_state' not in st.session_state:
    st.session_state['tracker_state'] = {}
if 'next_id' not in st.session_state:
    st.session_state['next_id'] = 1
    
# [新增] 手寫模式的結果記憶體 (防止跳掉)
if 'hw_display_list' not in st.session_state:
    st.session_state['hw_display_list'] = []
if 'hw_result_img' not in st.session_state:
    st.session_state['hw_result_img'] = None
if 'hw_result_count' not in st.session_state:
    st.session_state['hw_result_count'] = 0

with st.sidebar:
    st.title("🎛️ 控制台")
    app_mode = st.radio("模式選擇", ["📷 攝影機模式 (Live)", "🎨 手寫板模式"])
    st.divider()
    
    st.markdown("### 📷 鏡頭成績")
    c_total = st.session_state['stats']['camera']['total']
    c_correct = st.session_state['stats']['camera']['correct']
    c_acc = (c_correct / c_total * 100) if c_total > 0 else 0.0
    col_c1, col_c2 = st.columns(2)
    with col_c1: st.metric("總數", c_total)
    with col_c2: st.metric("正確", c_correct)
    st.metric("鏡頭準確率", f"{c_acc:.1f}%")
    
    col_undo_c, col_reset_c = st.columns(2)
    with col_undo_c:
        if st.button("↩️ 復原", key="undo_cam"):
            if st.session_state['history']['camera']:
                last_entry = st.session_state['history']['camera'].pop()
                st.session_state['stats']['camera']['total'] -= last_entry['total']
                st.session_state['stats']['camera']['correct'] -= last_entry['correct']
                st.rerun()
            else:
                st.toast("⚠️ 無紀錄可復原")
    with col_reset_c:
        if st.button("🗑️ 重置", key="reset_cam"):
            st.session_state['stats']['camera'] = {'total': 0, 'correct': 0}
            st.session_state['history']['camera'] = []
            st.rerun()

    st.divider()

    st.markdown("### 🎨 手寫成績")
    h_total = st.session_state['stats']['handwriting']['total']
    h_correct = st.session_state['stats']['handwriting']['correct']
    h_acc = (h_correct / h_total * 100) if h_total > 0 else 0.0
    col_h1, col_h2 = st.columns(2)
    with col_h1: st.metric("總數", h_total)
    with col_h2: st.metric("正確", h_correct)
    st.metric("手寫準確率", f"{h_acc:.1f}%")

    col_undo_h, col_reset_h = st.columns(2)
    with col_undo_h:
        if st.button("↩️ 復原", key="undo_hw"):
            if st.session_state['history']['handwriting']:
                last_entry = st.session_state['history']['handwriting'].pop()
                st.session_state['stats']['handwriting']['total'] -= last_entry['total']
                st.session_state['stats']['handwriting']['correct'] -= last_entry['correct']
                st.rerun()
            else:
                st.toast("⚠️ 無紀錄可復原")
    with col_reset_h:
        if st.button("🗑️ 重置", key="reset_hw"):
            # 修改處：只清除統計與追蹤器，保留顯示結果，避免畫面閃爍變空
            st.session_state['stats']['handwriting'] = {'total': 0, 'correct': 0}
            st.session_state['history']['handwriting'] = []
            st.session_state['tracker_state'] = {}
            st.session_state['next_id'] = 1
            # st.session_state['hw_display_list'] = []  <-- 移除這行
            # st.session_state['hw_result_img'] = None  <-- 移除這行
            # st.session_state['hw_result_count'] = 0   <-- 移除這行
            st.rerun()

st.title("📝 手寫數字辨識系統")

with st.expander("📖 系統操作說明 (點擊展開)", expanded=False):
    st.markdown("""
    #### 1. 📷 攝影機模式 (Live)
    * 點擊下方的 **START** 按鈕開啟攝影機。
    * 將手寫數字紙張置於藍色框框內。
    * **保持手部穩定**，下方進度條會開始跑，跑滿後畫面顯示 **"CAPTURED"** 並凍結。
    * 畫面凍結後，**點擊右側的「📋 顯示辨識詳情」按鈕**。
    * 確認後，點擊 **"💾 儲存並繼續"**。
    * 若抓拍結果不理想，可點擊 **"🔄 重新攝影"**。

    #### 2. 🎨 手寫板模式
    * 切換至左側選單的「手寫板模式」。
    * 直接在黑色畫布上用滑鼠書寫數字。
    * 放開滑鼠後，**左側下方**顯示詳細信心度，**右側**顯示綠框結果。
    """)

if model is None:
    st.error("❌ 找不到 `mnist_cnn.h5`！")
    st.stop()

# --- 5. 模式分支 ---

if app_mode == "📷 攝影機模式 (Live)":
    
    col_cam, col_data = st.columns([2, 1])

    with col_cam:
        ctx = webrtc_streamer(
            key="handwrite-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HandwriteProcessor,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

    with col_data:
        st.markdown("### 📊 詳細數據")
        st.caption("請等待畫面出現 Captured 後，按下方按鈕更新數據")
        
        # [修改] 使用兩欄來放置「顯示詳情」和「重新攝影」
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("📋 顯示詳情 (Update)", type="secondary", use_container_width=True):
                if ctx.video_processor and ctx.video_processor.frozen:
                    results = ctx.video_processor.ui_results
                    if results:
                        st.success(f"共偵測到 {len(results)} 個數字")
                        st.session_state['last_cam_detected'] = len(results)
                        for line in results:
                            st.markdown(line)
                    else:
                        st.warning("⚠️ 畫面凍結了，但沒有偵測到數字。")
                        st.session_state['last_cam_detected'] = 0
                else:
                    st.info("⏳ 請先等待鏡頭畫面抓拍凍結 (Captured)...")

        # [新增] 重新攝影按鈕
        with col_btn2:
            if st.button("🔄 重新攝影 (Retake)", type="primary", use_container_width=True):
                if ctx.video_processor:
                    ctx.video_processor.resume()
                # 重新刷新頁面以更新 UI 狀態
                st.session_state['last_cam_detected'] = 0
                st.rerun()

        st.write("---")
        
        manual_score = st.number_input("✍️ 輸入正確數量", min_value=0, value=0, key=f"score_input_{st.session_state['input_key']}")
        
        st.write("##") 
        if st.button("💾 儲存並繼續 (Save & Resume)", type="primary", use_container_width=True):
            
            total_add = st.session_state.get('last_cam_detected', 0)
            if total_add > 0 and manual_score > total_add:
                st.error(f"❌ 錯誤：輸入數值 ({manual_score}) 超過偵測總數 ({total_add})，多了 {manual_score - total_add} 個，請重新輸入！")
            else:
                if ctx.video_processor:
                    ctx.video_processor.resume()
                
                if total_add == 0: total_add = manual_score

                if manual_score > 0:
                    st.session_state['stats']['camera']['total'] += total_add
                    st.session_state['stats']['camera']['correct'] += manual_score
                    st.session_state['history']['camera'].append({
                        'total': total_add,
                        'correct': manual_score
                    })
                    st.toast(f"✅ 鏡頭模式：已記錄 (總數{total_add}/正確{manual_score})")
                    time.sleep(0.5)
                    st.session_state['input_key'] += 1
                    
                st.rerun()

elif app_mode == "🎨 手寫板模式":
    
    c_left, c_right = st.columns([3, 1])

    current_results_list = []
    
    with c_left:
        if st.button("🗑️ 清除畫布"):
            st.session_state['canvas_key'] = f"canvas_{time.time()}"
            st.session_state['tracker_state'] = {}
            st.session_state['next_id'] = 1
            # 修改處：清除畫布時，保留上一次的辨識結果在畫面上，直到寫下新字
            # st.session_state['hw_display_list'] = []  <-- 移除
            # st.session_state['hw_result_img'] = None  <-- 移除
            # st.session_state['hw_result_count'] = 0   <-- 移除
            st.rerun()

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,  
            width=850,   
            drawing_mode="freedraw",
            key=st.session_state['canvas_key'],
        )
        
        # 處理邏輯
        if canvas_result.image_data is not None:
            img_data = canvas_result.image_data.astype(np.uint8)
            
            if np.max(img_data) > 0:
                if img_data.shape[2] == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                binary_proc = cv2.dilate(gray, None, iterations=1)
                _, binary_proc = cv2.threshold(binary_proc, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                tracked_items = update_tracker(contours)
                
                draw_img = img_data.copy()
                batch_rois = []
                
                for item in tracked_items:
                    cnt = item['cnt']
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    if cv2.contourArea(cnt) > MIN_AREA:
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
                
                detected_count = 0
                if len(batch_rois) > 0:
                    preds = model.predict(np.stack(batch_rois), verbose=0)
                    for i, pred in enumerate(preds):
                        item = tracked_items[i]
                        display_id = item['id']
                        cnt = item['cnt']
                        
                        top_indices = pred.argsort()[-3:][::-1]
                        res_id = top_indices[0]
                        confidence = pred[res_id]
                        
                        x, y, w, h = cv2.boundingRect(cnt)
                        asp = w/h
                        if res_id==1 and asp>0.5: res_id=7
                        if res_id==7 and asp<0.3: res_id=1
                        
                        cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # [修改] 顯示辨識到的數字 (res_id)，字體稍微放大 (1.0)
                        cv2.putText(draw_img, str(res_id), (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        
                        display_text = f"**#{display_id}**: 數字 `{res_id}` (信心度: {int(confidence*100)}%)"
                        if confidence < 1.0: 
                            alts = []
                            for alt_idx in top_indices[1:]:
                                alt_conf = pred[alt_idx]
                                if alt_conf > 0.01:
                                    alts.append(f"`{alt_idx}` ({int(alt_conf*100)}%)")
                            if alts:
                                display_text += f" ⚠️ 其他: {', '.join(alts)}"
                        
                        current_results_list.append(display_text)
                        detected_count += 1
                
                # [關鍵] 更新全域記憶體 (圖片 & 數據)
                if current_results_list:
                    st.session_state['hw_display_list'] = current_results_list
                    st.session_state['hw_result_img'] = draw_img
                    st.session_state['hw_result_count'] = detected_count
            else:
                # 畫布變空時，不清空記憶，保留上一次結果，直到按清除
                pass

    with c_left:
        # 優先顯示記憶體中的資料
        if st.session_state['hw_display_list']:
            st.write("---")
            st.markdown("#### 📊 詳細數據:")
            cols = st.columns(2)
            for i, text in enumerate(st.session_state['hw_display_list']):
                cols[i % 2].markdown(text)

    with c_right:
        st.markdown("### 👁️ 結果")
        
        # 優先顯示記憶體中的圖片
        if st.session_state['hw_result_img'] is not None:
            st.image(st.session_state['hw_result_img'], channels="BGR", use_container_width=True)
        else:
            st.info("請在左側書寫")

        st.write("---")
        
        # 數量依據記憶體
        final_count = st.session_state['hw_result_count']

        if final_count > 0:
            st.success(f"偵測到: {final_count} 個")
        
        hw_score = st.number_input("輸入數量", min_value=0, value=final_count, key="hw_input")
        
        st.write("##")
        if st.button("💾 儲存", key="hw_save", type="primary"):
            
            # 手寫模式防呆
            if hw_score > final_count:
                st.error(f"❌ 錯誤：輸入數值 ({hw_score}) 超過偵測總數 ({final_count})，多了 {hw_score - final_count} 個，請重新輸入！")
            else:
                st.session_state['stats']['handwriting']['total'] += final_count
                st.session_state['stats']['handwriting']['correct'] += hw_score
                
                st.session_state['history']['handwriting'].append({
                    'total': final_count,
                    'correct': hw_score
                })
                
                # [新增] 儲存後，清空畫布與結果
                st.session_state['canvas_key'] = f"canvas_{time.time()}"
                st.session_state['tracker_state'] = {}
                st.session_state['next_id'] = 1
                st.session_state['hw_display_list'] = []
                st.session_state['hw_result_img'] = None
                st.session_state['hw_result_count'] = 0
                
                st.toast("✅ 手寫成績已儲存！")
                time.sleep(0.5)
                st.rerun()