import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas # 引入繪圖套件

# 設定 TensorFlow 日誌等級
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# --- 參數設定 ---
MIN_HEIGHT = 32
MIN_AREA = 140
SHRINK_PX = 4  # 視覺內縮

# --- 1. 頁面初始化與 Session State ---
st.set_page_config(page_title="手寫辨識 (旗艦版)", page_icon="📝", layout="wide")

# 初始化全域變數
if 'stats' not in st.session_state:
    st.session_state['stats'] = {'total': 0, 'correct': 0}
if 'last_photo' not in st.session_state:
    st.session_state['last_photo'] = None
if 'processed_image' not in st.session_state:
    st.session_state['processed_image'] = None
if 'detected_count' not in st.session_state:
    st.session_state['detected_count'] = 0
if 'input_locked' not in st.session_state:
    st.session_state['input_locked'] = False

# --- 2. 載入模型 ---
@st.cache_resource
def load_ai_model():
    if os.path.exists("mnist_cnn.h5"):
        try:
            return load_model("mnist_cnn.h5")
        except:
            return None
    return None

model = load_ai_model()

# --- 3. 核心功能函式 ---

def is_valid_content(img_bgr):
    """膚色/雜訊過濾器"""
    if img_bgr is None or img_bgr.size == 0: return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:,:,0]) 
    mean_s = np.mean(hsv[:,:,1]) 
    
    # 1. 飽和度過高 -> 雜物
    if mean_s > 60: return False 
    # 2. 飽和度中等且偏紅 -> 手部
    if 30 < mean_s <= 60:
        if (mean_h < 25 or mean_h > 155): return False 
    return True

def process_image(cv2_img, is_handwriting=False):
    """影像處理、CNN預測、混合修正、畫圖"""
    # 備份原圖用於畫框
    # 如果是手寫板傳來的 RGBA，先轉成 BGR
    if cv2_img.shape[2] == 4:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGBA2BGR)
        
    draw_img = cv2_img.copy()
    h_img, w_img = cv2_img.shape[:2]
    
    # 影像前處理
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    if is_handwriting:
        # 手寫模式已經是黑底白字，不需要太強的模糊與二值化
        # 直接取用 (稍微膨脹讓線條連貫)
        binary_proc = cv2.dilate(gray, None, iterations=1)
        # 確保真的夠黑白分明
        _, binary_proc = cv2.threshold(binary_proc, 127, 255, cv2.THRESH_BINARY)
    else:
        # 鏡頭模式：標準前處理
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 18)
        binary_proc = cv2.dilate(thresh, None, iterations=2)

    # 尋找輪廓
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

    # 準備批量預測
    batch_rois = []
    batch_info = []
    
    for item in valid_boxes:
        x, y, w, h = item["box"]
        
        # 鏡頭模式才需要過濾雜訊，手寫模式不用
        if not is_handwriting:
            if x < 15 or y < 15 or (x+w) > w_img-15 or (y+h) > h_img-15: continue
            if h < MIN_HEIGHT: continue
            # 膚色過濾
            roi_color = cv2_img[y:y+h, x:x+w]
            if not is_valid_content(roi_color): continue

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
            "coords": (x, y, w, h),
            "has_hole": item["has_hole"],
            "aspect": item["aspect_ratio"]
        })

    detected_count = 0
    
    # 執行預測
    if len(batch_rois) > 0 and model is not None:
        batch_input = np.stack(batch_rois)
        predictions = model.predict(batch_input, verbose=0)
        
        for i, pred in enumerate(predictions):
            res_id = np.argmax(pred)
            confidence = np.max(pred)
            info = batch_info[i]
            x, y, w, h = info["coords"]
            has_hole = info["has_hole"]
            aspect = info["aspect"]

            # === 混合修正邏輯 ===
            if res_id == 1:
                if aspect > 0.45: res_id = 7
            elif res_id == 7:
                if aspect < 0.25: res_id = 1
            if res_id == 7 and has_hole: res_id = 9
            if res_id == 9 and not has_hole and confidence < 0.95: res_id = 7
            if res_id == 0 and aspect < 0.5: res_id = 1
            # ===================

            # 視覺優化 (內縮框)
            draw_x = x + SHRINK_PX
            draw_y = y + SHRINK_PX
            draw_w = max(1, w - (SHRINK_PX * 2))
            draw_h = max(1, h - (SHRINK_PX * 2))

            cv2.rectangle(draw_img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), (0, 255, 0), 2)
            cv2.putText(draw_img, str(res_id), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            detected_count += 1
            
    return draw_img, detected_count

# --- 4. 介面佈局 ---

# 側邊欄
with st.sidebar:
    st.title("🎛️ 控制面板")
    
    # [新功能] 模式選擇
    app_mode = st.radio("選擇模式", ["📷 攝影機模式", "🎨 手寫板模式"])
    
    st.divider()
    
    # 統計數據
    total = st.session_state['stats']['total']
    correct = st.session_state['stats']['correct']
    acc = (correct / total * 100) if total > 0 else 0.0
    
    st.metric("累積總數", total)
    st.metric("累積正確", correct)
    st.metric("準確率", f"{acc:.1f}%")
    
    if st.button("🔄 重置統計"):
        st.session_state['stats'] = {'total': 0, 'correct': 0}
        st.rerun()

# 主畫面
st.title("📝 手寫數字辨識系統 (Web 旗艦版)")

if model is None:
    st.error("❌ 找不到 `mnist_cnn.h5`！請確認檔案位置。")
    st.stop()

# --- 5. 模式分支處理 ---

current_img = None
is_handwriting_mode = (app_mode == "🎨 手寫板模式")

if is_handwriting_mode:
    st.info("請在下方黑板直接書寫數字，放開滑鼠即自動辨識。")
    # 手寫板元件
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 填充色 (沒用到)
        stroke_width=15,                      # 筆刷粗細
        stroke_color="#FFFFFF",               # 筆刷顏色 (白)
        background_color="#000000",           # 背景顏色 (黑)
        height=300,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # 當畫布有內容時，進行處理
    if canvas_result.image_data is not None:
        # 轉換為 OpenCV 格式 (RGBA)
        img_data = canvas_result.image_data.astype(np.uint8)
        # 檢查是否全黑 (沒畫東西)
        if np.max(img_data) > 0:
            current_img = img_data

else:
    # 攝影機模式
    img_file_buffer = st.camera_input("📸 請對準數字，按下拍照按鈕進行辨識")
    if img_file_buffer is not None:
        # 檢查是否是新照片
        if img_file_buffer != st.session_state['last_photo']:
            bytes_data = img_file_buffer.getvalue()
            current_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            st.session_state['last_photo'] = img_file_buffer

# --- 6. 統一處理流程 ---

if current_img is not None:
    # 執行核心辨識
    processed_img, count = process_image(current_img, is_handwriting=is_handwriting_mode)
    
    # 更新顯示狀態
    st.session_state['processed_image'] = processed_img
    st.session_state['detected_count'] = count
    st.session_state['input_locked'] = False

    # 顯示結果
    st.image(st.session_state['processed_image'], channels="BGR", use_column_width=True)
    
    # 顯示檢測數量
    det_count = st.session_state['detected_count']
    if det_count > 0:
        if is_handwriting_mode:
            st.success(f"✨ 手寫板偵測到 **{det_count}** 個數字")
        else:
            st.info(f"🔍 畫面中偵測到 **{det_count}** 個數字")
    else:
        if not is_handwriting_mode:
            st.warning("⚠️ 未偵測到數字")

    # 成績輸入區
    st.write("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        manual_score = st.number_input(
            "✍️ 請輸入正確數字數量", 
            min_value=0, 
            max_value=det_count, 
            value=det_count,
            disabled=st.session_state['input_locked'],
            key=f"input_{time.time()}" # 強制更新 key 避免卡住
        )
    
    with col2:
        st.write("##") 
        if st.button("💾 確認並儲存", type="primary", disabled=st.session_state['input_locked']):
            if det_count > 0:
                st.session_state['stats']['total'] += det_count
                st.session_state['stats']['correct'] += manual_score
                st.session_state['input_locked'] = True
                
                success_msg = st.success("✅ 成績已儲存！")
                time.sleep(1)
                success_msg.empty()
                st.rerun()