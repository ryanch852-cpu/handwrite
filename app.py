import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# 設定 TensorFlow 日誌等級
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# --- 參數設定 (沿用之前的) ---
MIN_HEIGHT = 30
MIN_AREA = 140
SHRINK_PX = 4  # 視覺內縮

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="手寫數字辨識 (CNN Web版)", page_icon="📝")

st.title("📝 手寫數字辨識系統 (CNN + 混合修正)")
st.write("請使用攝影機拍攝手寫數字，系統將自動進行辨識。")

# --- 2. 載入模型 (使用 Cache 加速) ---
@st.cache_resource
def load_ai_model():
    if os.path.exists("mnist_cnn.h5"):
        return load_model("mnist_cnn.h5")
    return None

model = load_ai_model()

if model is None:
    st.error("❌ 找不到 `mnist_cnn.h5` 模型檔案！請確認檔案已上傳。")
    st.stop()

# --- 3. 輔助函式: 膚色過濾 ---
def is_valid_content(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:,:,0]) 
    mean_s = np.mean(hsv[:,:,1]) 
    
    # 飽和度過高 -> 雜物
    if mean_s > 60: return False 
    # 飽和度中等且偏紅 -> 手部
    if 30 < mean_s <= 60:
        if (mean_h < 25 or mean_h > 155): return False 
    return True

# --- 4. 攝影機輸入元件 ---
# Streamlit 的 camera_input 會直接讓你拍照並回傳圖片
img_file_buffer = st.camera_input("📸 點擊拍照進行辨識")

# --- 5. 核心處理邏輯 ---
if img_file_buffer is not None:
    # (1) 讀取影像
    bytes_data = img_file_buffer.getvalue()
    # 將 bytes 轉為 OpenCV 格式
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 建立一個畫圖用的乾淨複本
    result_img = cv2_img.copy()
    h_img, w_img = cv2_img.shape[:2]

    # (2) 影像前處理
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 18)
    binary_proc = cv2.dilate(thresh, None, iterations=2)

    # (3) 尋找輪廓
    contours, hierarchy = cv2.findContours(binary_proc, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1: # 最外層
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    has_hole = hierarchy[0][i][2] != -1
                    valid_boxes.append({
                        "box": (x, y, w, h), 
                        "has_hole": has_hole,
                        "aspect_ratio": w / float(h)
                    })

    # 由左至右排序
    valid_boxes = sorted(valid_boxes, key=lambda b: b["box"][0])

    # (4) 準備批量預測
    batch_rois = []
    batch_info = []
    
    for item in valid_boxes:
        x, y, w, h = item["box"]
        
        # 邊緣過濾
        if x < 10 or y < 10 or (x+w) > w_img-10 or (y+h) > h_img-10: continue
        if h < MIN_HEIGHT: continue

        # 膚色過濾
        roi_color = cv2_img[y:y+h, x:x+w]
        if not is_valid_content(roi_color): continue

        # CNN Preprocessing (Padding + Resize)
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

    # (5) 執行預測與顯示
    detected_count = 0
    results_text = []

    if len(batch_rois) > 0:
        batch_input = np.stack(batch_rois)
        predictions = model.predict(batch_input, verbose=0)
        
        for i, pred in enumerate(predictions):
            res_id = np.argmax(pred)
            confidence = np.max(pred)
            info = batch_info[i]
            x, y, w, h = info["coords"]
            has_hole = info["has_hole"]
            aspect = info["aspect"]

            # === 混合修正邏輯 (Hybrid Rules) ===
            if res_id == 1:
                if aspect > 0.45: res_id = 7
            elif res_id == 7:
                if aspect < 0.25: res_id = 1
            if res_id == 7 and has_hole: res_id = 9
            if res_id == 9 and not has_hole and confidence < 0.95: res_id = 7
            if res_id == 0 and aspect < 0.5: res_id = 1
            # =================================

            # 視覺優化 (內縮框)
            draw_x = x + SHRINK_PX
            draw_y = y + SHRINK_PX
            draw_w = max(1, w - (SHRINK_PX * 2))
            draw_h = max(1, h - (SHRINK_PX * 2))

            cv2.rectangle(result_img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), (0, 255, 0), 2)
            cv2.putText(result_img, str(res_id), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            detected_count += 1
            results_text.append(str(res_id))

    # (6) 顯示最終結果圖片
    st.image(result_img, channels="BGR", caption="辨識結果視圖")
    
    # 顯示統計資訊
    if detected_count > 0:
        st.success(f"✅ 成功辨識 {detected_count} 個數字！")
        st.info(f"辨識內容: {' '.join(results_text)}")
    else:
        st.warning("⚠️ 未偵測到數字，請調整距離或光線後重試。")

# --- 側邊欄說明 ---
with st.sidebar:
    st.header("使用說明")
    st.write("1. 允許瀏覽器使用攝影機")
    st.write("2. 將手寫數字對準鏡頭")
    st.write("3. 按下「拍照」按鈕")
    st.write("---")
    st.write("**功能特色:**")
    st.markdown("- CNN 深度學習辨識")
    st.markdown("- 混合規則修正 (1vs7, 7vs9)")
    st.markdown("- 膚色抗干擾過濾")