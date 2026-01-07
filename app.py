import streamlit as st
import cv2
import numpy as np
import os
import time
import av
import joblib  # 用於儲存/讀取 KNN 模型
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# 設定 TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist  # 用於訓練 KNN
from sklearn.neighbors import KNeighborsClassifier

# --- 參數設定 ---
# [距離控制]
MIN_HEIGHT = 50       
MIN_AREA = 500       

SHRINK_PX = 4
STABILITY_DURATION = 1.2
MOVEMENT_THRESHOLD = 80

# [過濾] 第一道防線：CNN 信心度門檻
CONFIDENCE_THRESHOLD = 0.85 

# [雙重驗證] 第二道防線：灰色地帶
KNN_VERIFY_RANGE = (0.85, 0.95)

# [設定] 藍框大小
ROI_MARGIN_X = 60   
ROI_MARGIN_Y = 60   
TEXT_Y_OFFSET = 15 

# --- 1. 模型載入與初始化 ---

@st.cache_resource
def load_ai_models():
    # 1. 載入 CNN
    cnn = None
    if os.path.exists("mnist_cnn.h5"):
        try:
            cnn = load_model("mnist_cnn.h5")
            print("✅ CNN 模型載入成功")
        except:
            print("❌ CNN 模型載入失敗")
    
    # 2. 載入或訓練 KNN (作為第二道防線)
    knn = None
    knn_path = "knn_model.pkl"
    
    if os.path.exists(knn_path):
        try:
            knn = joblib.load(knn_path)
            print("✅ KNN 模型載入成功")
        except:
            print("⚠️ KNN 模型損壞，重新訓練...")
    
    # 如果沒有 KNN 模型，現場訓練一個 (輕量版)
    if knn is None:
        print("⏳ 正在訓練 KNN 輔助模型 (僅需一次)...")
        try:
            (x_train, y_train), _ = mnist.load_data()
            x_flat = x_train.reshape(-1, 784) / 255.0
            
            # 為了啟動速度，只用前 10000 筆資料訓練
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_flat[:10000], y_train[:10000])
            
            joblib.dump(knn, knn_path)
            print("✅ KNN 模型訓練完成並儲存")
        except Exception as e:
            print(f"❌ KNN 訓練失敗: {e}")
            knn = None

    return cnn, knn

model, knn_model = load_ai_models()

# --- [自動扶正] Deskewing ---
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# --- 2. 核心檢測功能 ---
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
        self.knn = knn_model
        self.last_boxes = []
        self.stability_start_time = None
        self.frozen = False        
        self.frozen_frame = None  
        self.detected_count = 0   
        self.ui_results = [] 
        
        self.frame_counter = 0
        self.skip_rate = 4  
        self.cached_rois = [] 

    def resume(self):
        self.frozen = False
        self.stability_start_time = None
        self.last_boxes = []
        self.ui_results = [] 
        self.frame_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.frozen and self.frozen_frame is not None:
            return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
        
        display_img = img.copy()
        h_f, w_f = img.shape[:2]
        
        roi_rect = [ROI_MARGIN_X, ROI_MARGIN_Y, w_f - 2*ROI_MARGIN_X, h_f - 2*ROI_MARGIN_Y]
        cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), 
                      (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (255, 0, 0), 2)

        self.frame_counter += 1
        process_this_frame = (self.frame_counter % self.skip_rate == 0)

        if not process_this_frame and len(self.cached_rois) > 0:
            for (dx, dy, dw, dh, txt, box_color) in self.cached_rois:
                cv2.rectangle(display_img, (dx, dy), (dx+dw, dy+dh), box_color, 2)
                cv2.putText(display_img, txt, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")
        
        # --- 處理邏輯 ---
        roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
        if roi_img.size == 0: return av.VideoFrame.from_ndarray(display_img, format="bgr24")

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
        
        self.cached_rois = []

        for item in valid_boxes:
            x, y, w, h = item["box"]
            rx, ry = x + roi_rect[0], y + roi_rect[1]
            
            if x < 5 or y < 5 or (x+w) > binary_proc.shape[1]-5 or (y+h) > binary_proc.shape[0]-5: continue
            if h < MIN_HEIGHT: continue
            
            roi_color = display_img[ry:ry+h, rx:rx+w]
            if not is_valid_content(roi_color): continue
            
            raw_boxes_for_stability.append(item)
            
            roi_single = binary_proc[y:y+h, x:x+w]
            roi_single = deskew(roi_single)

            side = max(w, h)
            padding = int(side * 0.2)
            container_size = side + padding * 2
            container = np.zeros((container_size, container_size), dtype=np.uint8)
            offset_y = (container_size - h) // 2
            offset_x = (container_size - w) // 2
            
            roi_single = cv2.resize(roi_single, (w, h)) 
            container[offset_y:offset_y+h, offset_x:offset_x+w] = roi_single
            roi_resized = cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA)
            
            roi_norm = roi_resized.astype('float32') / 255.0
            roi_ready = roi_norm.reshape(28, 28, 1)
            
            batch_rois.append(roi_ready)
            batch_info.append({
                "coords": (rx, ry, w, h),
                "has_hole": item["has_hole"],
                "aspect": item["aspect_ratio"],
                "flat_input": roi_norm.reshape(1, 784) # 用於 KNN
            })
            
        detected_count = 0
        detected_something = False
        current_frame_text_results = []
        
        # [修改] 新增一個計數器，用於顯示連續的序號
        valid_ui_counter = 1

        if len(batch_rois) > 0 and self.model is not None:
            detected_something = True
            try:
                batch_input = np.stack(batch_rois)
                predictions = self.model.predict(batch_input, verbose=0)
                
                for i, pred in enumerate(predictions):
                    top_indices = pred.argsort()[-3:][::-1]
                    res_id = top_indices[0]
                    confidence = pred[res_id]
                    
                    if confidence < CONFIDENCE_THRESHOLD: continue 

                    info = batch_info[i]
                    rx, ry, w, h = info["coords"]
                    has_hole = info["has_hole"]
                    aspect = info["aspect"]
                    
                    # 邏輯判斷
                    if res_id == 1 and aspect > 0.6: res_id = 7
                    elif res_id == 7 and aspect < 0.25: res_id = 1
                    if res_id == 7 and has_hole: res_id = 9
                    if res_id == 9 and not has_hole and confidence < 0.95: res_id = 7
                    if res_id == 0 and aspect < 0.5: res_id = 1
                    
                    # --- [KNN 雙重驗證] ---
                    final_label_str = str(res_id)
                    box_color = (0, 255, 0) # 預設綠色
                    verify_msg = ""
                    
                    if self.knn is not None and KNN_VERIFY_RANGE[0] <= confidence <= KNN_VERIFY_RANGE[1]:
                        try:
                            knn_pred = self.knn.predict(info["flat_input"])[0]
                            if knn_pred != res_id:
                                final_label_str = str(res_id) 
                                verify_msg = f" ⚠️ KNN: {knn_pred}"
                                box_color = (0, 165, 255) # 橘色表示有疑慮
                        except:
                            pass
                    # ----------------------------
                    
                    draw_x = rx + SHRINK_PX
                    draw_y = ry + SHRINK_PX
                    draw_w = max(1, w - (SHRINK_PX * 2))
                    draw_h = max(1, h - (SHRINK_PX * 2))
                    
                    cv2.rectangle(display_img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), box_color, 2)
                    
                    # [修改] 使用 valid_ui_counter 來顯示序號，而不是 i+1
                    text_label = f"#{valid_ui_counter}"
                    cv2.putText(display_img, text_label, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    self.cached_rois.append((draw_x, draw_y, draw_w, draw_h, text_label, box_color))
                    
                    # UI 文字 (Camera 模式用純文字)
                    # [修改] 同步 UI 文字使用新的連續序號
                    info_text = f"**#{valid_ui_counter}**: 數字 `{res_id}` (信心: {int(confidence*100)}%){verify_msg}"
                    
                    if confidence < 1.0 and "KNN" not in verify_msg:
                        alt_id = top_indices[1]
                        alt_conf = pred[alt_id]
                        if alt_conf > 0.01:
                            info_text += f" <span style='color:gray'>(次選: {alt_id})</span>"
                            
                    current_frame_text_results.append(info_text)
                    
                    detected_count += 1
                    valid_ui_counter += 1 # 只有真正顯示時才 +1
                    
            except Exception as e: 
                print(e)
                pass

        self.detected_count = detected_count
        if detected_something:
             self.ui_results = current_frame_text_results

        # Stability 邏輯 (省略細節以節省版面，保持不變)
        if len(raw_boxes_for_stability) == 0:
            self.stability_start_time = None
        elif len(self.last_boxes) == 0:
            self.last_boxes = raw_boxes_for_stability
            self.stability_start_time = time.time()
        else:
            total_movement = 0
            for curr_box in raw_boxes_for_stability:
                c_x, c_y, _, _ = curr_box["box"]
                min_dist = 99999
                for last_box in self.last_boxes:
                    l_x, l_y, _, _ = last_box["box"]
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
                    self.frozen_frame = display_img.copy()
            else:
                self.stability_start_time = time.time()

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# --- 4. Streamlit 介面 ---
st.set_page_config(page_title="手寫辨識", page_icon="📝", layout="wide")

if 'stats' not in st.session_state:
    st.session_state['stats'] = {
        'camera': {'total': 0, 'correct': 0}, 
        'handwriting': {'total': 0, 'correct': 0},
        'upload': {'total': 0, 'correct': 0} 
    }
if 'history' not in st.session_state:
    st.session_state['history'] = {'camera': [], 'handwriting': [], 'upload': []} 
    
if 'input_key' not in st.session_state: st.session_state['input_key'] = 0
if 'canvas_key' not in st.session_state: st.session_state['canvas_key'] = "canvas_0"
if 'tracker_state' not in st.session_state: st.session_state['tracker_state'] = {}
if 'next_id' not in st.session_state: st.session_state['next_id'] = 1
    
if 'hw_display_list' not in st.session_state: st.session_state['hw_display_list'] = []
if 'hw_result_img' not in st.session_state: st.session_state['hw_result_img'] = None
if 'hw_result_count' not in st.session_state: st.session_state['hw_result_count'] = 0

if 'upload_display_list' not in st.session_state: st.session_state['upload_display_list'] = []
if 'upload_result_img' not in st.session_state: st.session_state['upload_result_img'] = None
if 'upload_result_count' not in st.session_state: st.session_state['upload_result_count'] = 0
if 'last_uploaded_file_id' not in st.session_state: st.session_state['last_uploaded_file_id'] = None

with st.sidebar:
    st.title("🎛️ 控制台")
    app_mode = st.radio("模式選擇", ["📷 攝影機模式 (Live)", "🎨 手寫板模式", "📁 圖片上傳模式"], index=1)
    
    st.divider()
    
    # --- 成績區塊 ---
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
    with col_reset_h:
        if st.button("🗑️ 重置", key="reset_hw"):
            st.session_state['stats']['handwriting'] = {'total': 0, 'correct': 0}
            st.session_state['history']['handwriting'] = []
            st.session_state['tracker_state'] = {}
            st.session_state['next_id'] = 1
            st.rerun()

    st.divider()

    st.markdown("### 📁 上傳成績")
    u_total = st.session_state['stats']['upload']['total']
    u_correct = st.session_state['stats']['upload']['correct']
    u_acc = (u_correct / u_total * 100) if u_total > 0 else 0.0
    col_u1, col_u2 = st.columns(2)
    with col_u1: st.metric("總數", u_total)
    with col_u2: st.metric("正確", u_correct)
    st.metric("上傳準確率", f"{u_acc:.1f}%")

    col_undo_u, col_reset_u = st.columns(2)
    with col_undo_u:
        if st.button("↩️ 復原", key="undo_up"):
            if st.session_state['history']['upload']:
                last_entry = st.session_state['history']['upload'].pop()
                st.session_state['stats']['upload']['total'] -= last_entry['total']
                st.session_state['stats']['upload']['correct'] -= last_entry['correct']
                st.rerun()
    with col_reset_u:
        if st.button("🗑️ 重置", key="reset_up"):
            st.session_state['stats']['upload'] = {'total': 0, 'correct': 0}
            st.session_state['history']['upload'] = []
            st.session_state['upload_display_list'] = []
            st.session_state['upload_result_img'] = None
            st.session_state['upload_result_count'] = 0
            st.rerun()

st.title("📝 手寫數字辨識系統")

with st.expander("📖 系統操作說明 (點擊展開)", expanded=False):
    st.markdown(f"""
    #### ⚠️ 提高準確率的技巧：
    1. 手寫模式中如果發現沒出現綠色框，代表信心度過低或沒判定到，可以考慮把字寫整齊
    2. 鏡頭模式中請將紙張拿近鏡頭，數字太小會被忽略，筆跡太細也會被忽略，盡量拿奇異筆寫。
    3. 數字**1**不要畫底線，會被判定成其他數字。
    4. 數字盡量寫正。
    5. 鏡頭模式中顯示的是序號，實際數值請點選📋 顯示詳情查看
    > **注意**：系統設定信心度低於 **{int(CONFIDENCE_THRESHOLD*100)}%** 的結果將不會顯示。
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
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

    with col_data:
        st.markdown("### 📊 詳細數據")
        st.caption("請等待畫面出現 Captured 後，按下方按鈕更新數據")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("📋 顯示詳情", type="secondary", use_container_width=True):
                if ctx.video_processor and ctx.video_processor.frozen:
                    results = ctx.video_processor.ui_results
                    if results:
                        st.success(f"共偵測到 {len(results)} 個數字")
                        st.session_state['last_cam_detected'] = len(results)
                        for line in results:
                            st.markdown(line, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ 畫面凍結了，但沒有偵測到數字。")
                        st.session_state['last_cam_detected'] = 0
                else:
                    st.info("⏳ 請先等待鏡頭畫面抓拍凍結 (Captured)...")

        with col_btn2:
            if st.button("🔄 重新攝影", type="primary", use_container_width=True):
                if ctx.video_processor:
                    ctx.video_processor.resume()
                st.session_state['last_cam_detected'] = 0
                st.rerun()

        st.write("---")
        
        manual_score = st.number_input("✍️ 輸入正確數量", min_value=0, value=0, key=f"score_input_{st.session_state['input_key']}")
        
        st.write("##") 
        if st.button("💾 上傳成績並繼續", type="primary", use_container_width=True):
            
            total_add = st.session_state.get('last_cam_detected', 0)
            if total_add > 0 and manual_score > total_add:
                st.error(f"❌ 錯誤：輸入數值 ({manual_score}) 超過偵測總數 ({total_add})")
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

# ... (前面的程式碼保持不變)

elif app_mode == "🎨 手寫板模式":
    
    # [輔助函式] 顯示信心度條
    def get_bar_html(confidence, is_uncertain=False):
        percent = min(int(confidence * 100), 100)
        if is_uncertain: color = "#ff9f43" 
        elif confidence > 0.95: color = "#2ecc71"
        elif confidence > 0.85: color = "#f1c40f"
        else: color = "#e74c3c"
        
        return f"""
        <div style="display: flex; align-items: center; margin-top: 4px;">
            <div style="width: 50%; height: 8px; background-color: #444; border-radius: 4px; overflow: hidden;">
                <div style="width: {percent}%; height: 100%; background-color: {color};"></div>
            </div>
            <span style="margin-left: 8px; font-size: 0.8em; color: {color};">{percent}%</span>
        </div>
        """

    # --- 版面配置 ---
    # ... (在 elif app_mode == "🎨 手寫板模式": 裡面) ...

    c_left, c_right = st.columns([3, 2])
    
    with c_right:
        st.markdown("### 👁️ 結果")
        result_image_placeholder = st.empty()
        
        # 顯示圖片的邏輯 (之前改過的黑色空圖邏輯)
        if st.session_state['hw_result_img'] is not None:
             result_image_placeholder.image(st.session_state['hw_result_img'], channels="BGR", use_container_width=True)
        else:
             blank_img = np.zeros((400, 600, 3), dtype=np.uint8)
             cv2.putText(blank_img, "Waiting...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
             result_image_placeholder.image(blank_img, channels="BGR", use_container_width=True, caption="請在左側書寫")

        st.write("---")
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # [重要] 這裡一定要建立這個佔位符，輸入框才會出現在這裡！
        result_stats_placeholder = st.empty()
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    current_results_list = []
    
    # 2. 左側畫布與邏輯
    with c_left:
        if st.button("🗑️ 清除畫布"):
            st.session_state['canvas_key'] = f"canvas_{time.time()}"
            st.session_state['tracker_state'] = {}
            st.session_state['next_id'] = 1
            st.session_state['hw_display_list'] = [] 
            st.session_state['hw_result_img'] = None
            st.session_state['hw_result_count'] = 0
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
            display_toolbar=False,
            update_streamlit=True, 
        )
        
        # --- 核心處理邏輯 (保持原本邏輯不變) ---
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
                flat_inputs = [] 
                
                for item in tracked_items:
                    cnt = item['cnt']
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    if cv2.contourArea(cnt) > MIN_AREA:
                        roi = binary_proc[y:y+h, x:x+w]
                        roi = deskew(roi) 
                        
                        side = max(w, h)
                        pad = int(side * 0.2)
                        container = np.zeros((side+pad*2, side+pad*2), dtype=np.uint8)
                        ox, oy = (side+pad*2-w)//2, (side+pad*2-h)//2
                        
                        roi = cv2.resize(roi, (w, h))
                        container[oy:oy+h, ox:ox+w] = roi
                        
                        roi_ready = cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA)
                        roi_norm = roi_ready.astype('float32') / 255.0
                        
                        batch_rois.append(roi_norm.reshape(28, 28, 1))
                        flat_inputs.append(roi_norm.reshape(1, 784))
                
                detected_count = 0
                valid_ui_counter = 1

                if len(batch_rois) > 0:
                    preds = model.predict(np.stack(batch_rois), verbose=0)
                    
                    for i, pred in enumerate(preds):
                        item = tracked_items[i]
                        cnt = item['cnt']
                        
                        top_indices = pred.argsort()[-3:][::-1]
                        res_id = top_indices[0]
                        confidence = pred[res_id]
                        
                        if confidence < CONFIDENCE_THRESHOLD:
                            continue

                        x, y, w, h = cv2.boundingRect(cnt)
                        asp = w/h
                        
                        if res_id==1 and asp>0.6: res_id=7 
                        if res_id==7 and asp<0.3: res_id=1
                        
                        is_uncertain = False
                        verify_text_html = ""
                        final_res = str(res_id)
                        box_color = (0, 255, 0)
                        
                        if knn_model is not None and KNN_VERIFY_RANGE[0] <= confidence <= KNN_VERIFY_RANGE[1]:
                            try:
                                k_pred = knn_model.predict(flat_inputs[i])[0]
                                if k_pred != res_id:
                                    is_uncertain = True
                                    verify_text_html = f"<div style='color:#ff9f43; font-size:0.85em; margin-bottom: 2px;'>⚠️ KNN 建議: {k_pred}</div>"
                                    final_res = str(res_id)
                                    box_color = (0, 165, 255)
                            except: pass
                        
                        cv2.rectangle(draw_img, (x, y), (x+w, y+h), box_color, 2)
                        cv2.putText(draw_img, final_res, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        
                        text_part = f"<div>#{valid_ui_counter}: 數字 <strong>{res_id}</strong></div>"
                        if is_uncertain: text_part += verify_text_html
                        elif confidence < 1.0: 
                            alts = []
                            for alt_idx in top_indices[1:]:
                                if pred[alt_idx] > 0.01: alts.append(f"{alt_idx}({int(pred[alt_idx]*100)}%)")
                            if alts: text_part += f"<div style='color:gray; font-size:0.8em'>⚠️ 其他: {', '.join(alts)}</div>"
                        
                        bar_part = get_bar_html(confidence, is_uncertain)
                        current_results_list.append(f"<div style='margin-bottom:10px;'>{text_part}{bar_part}</div>")
                        
                        detected_count += 1
                        valid_ui_counter += 1

                # 更新圖片與狀態
                result_image_placeholder.image(draw_img, channels="BGR", use_container_width=True)
                
                st.session_state['hw_display_list'] = current_results_list
                st.session_state['hw_result_img'] = draw_img
                st.session_state['hw_result_count'] = detected_count

    # 3. 顯示下方的詳細數據
    with c_left:
        if st.session_state['hw_display_list']:
            st.write("---")
            st.markdown("#### 📊 詳細數據:")
            cols = st.columns(2)
            for i, html_content in enumerate(st.session_state['hw_display_list']):
                cols[i % 2].markdown(html_content, unsafe_allow_html=True)

    status_placeholder = st.empty()
    
    final_count = st.session_state['hw_result_count']
    
    wrapper_style = "min-height: 60px; margin-bottom: 10px;"
    
    if final_count > 0:
        # 綠色狀態
        status_html = f"""
        <div style="{wrapper_style}">
            <div style="
                padding: 10px;
                border-radius: 5px;
                background-color: #d1e7dd; 
                color: #0f5132;
                border: 1px solid #badbcc;">
                ✅ 偵測到: <strong>{final_count}</strong> 個
            </div>
        </div>
        """
    else:
        # 藍色狀態 (佔位)
        status_html = f"""
        <div style="{wrapper_style}">
            <div style="
                padding: 10px;
                border-radius: 5px;
                background-color: #cff4fc;
                color: #055160;
                border: 1px solid #b6effb;">
                ℹ️ 等待書寫中...
            </div>
        </div>
        """
        
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    # [關鍵] 輸入框放在 placeholder 外面！
    # 這樣 status_placeholder 更新時，這個輸入框就不會被銷毀重蓋
    hw_score = st.number_input("輸入數量", min_value=0, value=final_count, key="hw_input")
    
    st.write("##")
    if st.button("💾 上傳成績", key="hw_save", type="primary"):
        if hw_score > final_count:
            st.error(f"❌ 錯誤：輸入數值 ({hw_score}) 超過偵測總數 ({final_count})")
        else:
            st.session_state['stats']['handwriting']['total'] += final_count
            st.session_state['stats']['handwriting']['correct'] += hw_score
            st.session_state['history']['handwriting'].append({'total': final_count, 'correct': hw_score})
            
            # 重置狀態
            st.session_state['canvas_key'] = f"canvas_{time.time()}"
            st.session_state['tracker_state'] = {}
            st.session_state['next_id'] = 1
            st.session_state['hw_display_list'] = []
            st.session_state['hw_result_img'] = None
            st.session_state['hw_result_count'] = 0
            
            if 'hw_input' in st.session_state:
                del st.session_state['hw_input']
            
            st.toast("✅ 手寫成績已儲存！")
            time.sleep(0.5)
            st.rerun()

elif app_mode == "📁 圖片上傳模式":

    # --- 1. 來源判斷 ---
    def detect_image_source(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return "digital" if (np.sum(gray > 250) / gray.size) > 0.3 else "photo"

    # --- 2. 物理融合 ---
    def merge_overlapping_boxes(boxes):
        if len(boxes) < 2: return boxes
        merged = []
        while len(boxes) > 0:
            curr = boxes.pop(0)
            x1, y1, w1, h1 = curr
            rx1, ry1 = x1 + w1, y1 + h1
            has_overlap = False
            i = 0
            while i < len(boxes):
                next_box = boxes[i]
                x2, y2, w2, h2 = next_box
                rx2, ry2 = x2 + w2, y2 + h2
                pad = 15
                overlap = not ((rx1 + pad) < x2 or (x1 - pad) > rx2 or (ry1 + pad) < y2 or (y1 - pad) > ry2)
                if overlap:
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(rx1, rx2) - new_x
                    new_h = max(ry1, ry2) - new_y
                    curr = (new_x, new_y, new_w, new_h)
                    x1, y1, w1, h1 = curr
                    rx1, ry1 = new_x + new_w, new_y + new_h
                    boxes.pop(i)
                    has_overlap = True
                else:
                    i += 1
            if has_overlap:
                boxes.insert(0, curr)
            else:
                merged.append(curr)
        return merged

    # --- 3. [分流修正] 尺寸過濾器 (數位寬鬆，手機嚴格) ---
    def filter_small_boxes(boxes, img_height, source_type):
        if not boxes: return []
        
        # 1. 數位模式：極度寬鬆 (保護 2)
        if source_type == "digital":
            kept = []
            for box in boxes:
                # 只要不是奈米級雜點 (h>15) 就保留
                if box[3] > 15: kept.append(box)
            return kept

        # 2. 手機模式：嚴格過濾 (殺污漬)
        
        # 絕對底線 (2%)
        abs_min_h = int(img_height * 0.02)
        
        # 計算中位數 (只用有效框)
        valid_h = [b[3] for b in boxes if b[3] > abs_min_h]
        valid_area = [b[2]*b[3] for b in boxes if b[3] > abs_min_h]
        
        median_h = np.median(valid_h) if valid_h else 0
        median_area = np.median(valid_area) if valid_area else 0
        
        kept_boxes = []
        for box in boxes:
            w, h = box[2], box[3]
            area = w * h
            aspect = w / float(h)
            
            # [規則 A] 絕對底線
            if h < abs_min_h: continue
            
            # [規則 B] 瘦子保護 (針對 1)
            if aspect < 0.35:
                # 瘦子只要有 35% 平均身高就過
                if median_h > 0 and h > (median_h * 0.35):
                    kept_boxes.append(box)
                continue
            
            # [規則 C] 一般物件 (針對 0, 2, 污漬)
            # 1. 身高必須達到中位數的 50%
            if median_h > 0 and h < (median_h * 0.5):
                continue
                
            # 2. 面積必須達到中位數的 20% (殺小圓點)
            if median_area > 0 and area < (median_area * 0.2):
                continue

            kept_boxes.append(box)
            
        return kept_boxes

    # --- 4. 墨水濃度過濾 ---
    def filter_low_contrast_boxes(boxes, gray_img):
        if not boxes: return []
        flat = np.sort(gray_img.ravel())
        ink_black = np.mean(flat[:int(len(flat)*0.02)])
        paper_bg = np.median(flat)
        dynamic_range = paper_bg - ink_black
        threshold = paper_bg - (dynamic_range * 0.6)
        
        kept_boxes = []
        for box in boxes:
            x, y, w, h = box
            roi = gray_img[y:y+h, x:x+w]
            if roi.size == 0: continue
            roi_flat = np.sort(roi.ravel())
            roi_darkest = np.mean(roi_flat[:max(1, int(len(roi_flat)*0.1))])
            if roi_darkest > threshold: continue
            kept_boxes.append(box)
        return kept_boxes

    # --- 5. MNIST 標準化 ---
    def preprocess_for_mnist(roi_binary):
        h, w = roi_binary.shape
        canvas = np.zeros((28, 28), dtype=np.uint8)
        scale = 20.0 / max(h, w)
        nh = max(1, int(h * scale))
        nw = max(1, int(w * scale))
        roi_resized = cv2.resize(roi_binary, (nw, nh), interpolation=cv2.INTER_AREA)
        y_off = (28 - nh) // 2
        x_off = (28 - nw) // 2
        y_end = min(y_off + nh, 28)
        x_end = min(x_off + nw, 28)
        canvas[y_off:y_end, x_off:x_end] = roi_resized[:y_end-y_off, :x_end-x_off]
        
        _, canvas = cv2.threshold(canvas, 10, 255, cv2.THRESH_BINARY)
        
        M = cv2.moments(canvas)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            shift_x = 14 - cx
            shift_y = 14 - cy
            M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, M_shift, (28, 28))
        canvas = cv2.dilate(canvas, None, iterations=1)
        return canvas

    # --- 信心度條 ---
    def get_bar_html(confidence, is_uncertain=False):
        percent = min(int(confidence * 100), 100)
        color = "#e74c3c"
        if is_uncertain: color = "#ff9f43"
        elif confidence > 0.95: color = "#2ecc71"
        elif confidence > 0.85: color = "#f1c40f"
        return f"""
        <div style="display: flex; align-items: center; margin-top: 4px;">
            <div style="width: 50%; height: 8px; background-color: #444; border-radius: 4px; overflow: hidden;">
                <div style="width: {percent}%; height: 100%; background-color: {color};"></div>
            </div>
            <span style="margin-left: 8px; font-size: 0.8em; color: {color};">{percent}%</span>
        </div>
        """

    col_up_left, col_up_right = st.columns([3, 1])
    
    with col_up_left:
        uploaded_file = st.file_uploader("請上傳圖片 (JPG, PNG)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            if st.session_state['last_uploaded_file_id'] != uploaded_file.file_id:
                st.session_state['last_uploaded_file_id'] = uploaded_file.file_id
                
                source_type = detect_image_source(img)
                display_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 前處理
                if source_type == "photo":
                    st.info("📸 模式：手機翻拍 (嚴格除垢)")
                    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
                    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 45, 12)
                    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    binary_proc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect)
                    min_area_limit = 10 
                else:
                    st.success("💻 模式：數位截圖")
                    _, binary_proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    binary_proc = cv2.morphologyEx(binary_proc, cv2.MORPH_CLOSE, kernel_connect)
                    min_area_limit = 30

                with st.expander("👀 Debug: 機器看到的畫面"):
                    st.image(binary_proc, caption="二值化結果", clamp=True, channels='GRAY')

                contours, hierarchy = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                raw_boxes = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    if area < min_area_limit: continue
                    if h < 5: continue 
                    raw_boxes.append((x, y, w, h))

                merged_boxes = merge_overlapping_boxes(raw_boxes)
                
                # [關鍵] 分流過濾 (傳入 source_type)
                h_img_total = img.shape[0]
                sized_boxes = filter_small_boxes(merged_boxes, h_img_total, source_type)
                
                # 只有照片才需要墨水過濾
                final_boxes = sized_boxes
                if source_type == "photo":
                    final_boxes = filter_low_contrast_boxes(sized_boxes, gray)

                batch_rois = []
                batch_info = []
                
                for (x, y, w, h) in final_boxes:
                    roi = binary_proc[y:y+h, x:x+w]
                    
                    if source_type == "photo" and h < 150:
                        roi = deskew(roi)
                    
                    final_norm = preprocess_for_mnist(roi)
                    final_input = final_norm.astype('float32') / 255.0
                    
                    has_hole = False
                    roi_u8 = (final_input * 255).astype(np.uint8)
                    _, t_roi = cv2.threshold(roi_u8, 50, 255, cv2.THRESH_BINARY)
                    c_sub, h_sub = cv2.findContours(t_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    if h_sub is not None:
                        for idx, cc in enumerate(c_sub):
                            if h_sub[0][idx][3] != -1 and cv2.contourArea(cc) > 5:
                                has_hole = True
                                break

                    batch_rois.append(final_input.reshape(28, 28, 1))
                    batch_info.append({
                        "rect": (x, y, w, h),
                        "has_hole": has_hole,
                        "aspect": w / float(h),
                        "flat_input": final_input.reshape(1, 784)
                    })

                detected_count = 0
                results_text = []
                valid_ui_counter = 1
                
                h_disp, w_disp = display_img.shape[:2]
                scale = max(1.0, w_disp / 800.0)
                font_s = 1.0 * scale
                thick = max(2, int(3 * scale))

                if len(batch_rois) > 0:
                    combined = list(zip(batch_rois, batch_info))
                    combined.sort(key=lambda x: x[1]["rect"][0])
                    
                    sorted_rois = [x[0] for x in combined]
                    sorted_info = [x[1] for x in combined]
                    
                    predictions = model.predict(np.stack(sorted_rois), verbose=0)
                    
                    for i, pred in enumerate(predictions):
                        top_indices = pred.argsort()[-3:][::-1]
                        res_id = top_indices[0]
                        confidence = pred[res_id]
                        
                        info = sorted_info[i]
                        x, y, w, h = info["rect"]
                        has_hole = info["has_hole"]
                        aspect = info["aspect"]

                        thresh = CONFIDENCE_THRESHOLD
                        if h > 150: thresh = 0.5
                        if confidence < thresh: continue

                        if res_id == 7 and aspect < 0.25: res_id = 1
                        if res_id == 1 and has_hole: res_id = 0
                        if source_type == "digital" and aspect < 0.2: res_id = 1
                        
                        color = (0, 255, 0)
                        extra_msg = ""
                        
                        if knn_model is not None and KNN_VERIFY_RANGE[0] <= confidence <= 0.99:
                             try:
                                k_res = knn_model.predict(info["flat_input"])[0]
                                if k_res != res_id:
                                    extra_msg = f" (KNN: {k_res})"
                                    if res_id == 8 and k_res == 9 and has_hole:
                                        res_id = 9
                                        color = (0, 165, 255)
                             except: pass

                        cv2.rectangle(display_img, (x, y), (x+w, y+h), color, thick)
                        cv2.putText(display_img, str(res_id), (x, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_s, (0, 0, 255), thick)

                        text_html = f"<div><strong>#{valid_ui_counter}</strong>: {res_id} <span style='font-size:0.8em; color:gray'>{extra_msg}</span></div>"
                        bar_html = get_bar_html(confidence)
                        results_text.append(f"<div style='margin-bottom:8px'>{text_html}{bar_html}</div>")
                        
                        detected_count += 1
                        valid_ui_counter += 1

                st.session_state['upload_result_img'] = display_img
                st.session_state['upload_display_list'] = results_text
                st.session_state['upload_result_count'] = detected_count

            if st.session_state['upload_result_img'] is not None:
                st.image(st.session_state['upload_result_img'], channels="BGR", use_container_width=True)
            
            if st.session_state['upload_display_list']:
                st.divider()
                st.markdown("#### 📊 辨識結果")
                cols = st.columns(3)
                for idx, txt in enumerate(st.session_state['upload_display_list']):
                    cols[idx % 3].markdown(txt, unsafe_allow_html=True)

    with col_up_right:
        st.markdown("### 📝 確認")
        final_cnt = st.session_state['upload_result_count']
        
        if final_cnt > 0:
            st.success(f"偵測到 {final_cnt} 個")
        else:
            if uploaded_file: st.warning("未偵測到")
            
        real_val = st.number_input("正確數量", min_value=0, value=final_cnt, key="up_input_val")
        
        st.write("##")
        if st.button("💾 儲存", type="primary", use_container_width=True):
            if final_cnt == 0 and real_val == 0:
                st.toast("無資料")
            else:
                save_val = final_cnt if final_cnt > 0 else real_val
                st.session_state['stats']['upload']['total'] += save_val
                st.session_state['stats']['upload']['correct'] += real_val
                st.session_state['history']['upload'].append({'total': save_val, 'correct': real_val})
                st.toast("✅ 已儲存")
                st.session_state['upload_result_img'] = None
                st.session_state['upload_display_list'] = []
                st.session_state['upload_result_count'] = 0
                st.session_state['last_uploaded_file_id'] = None
                time.sleep(0.5)
                st.rerun()