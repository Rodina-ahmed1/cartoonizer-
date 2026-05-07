from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

def apply_heavy_cartoon(img_rgb, k=8, smoothing_passes=10, thickness=1):
    if img_rgb.shape[-1] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    for _ in range(smoothing_passes):
        l = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75)
    smooth_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    h, w = smooth_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    is_color_photo = hsv[:, :, 1].mean() > 15
    if is_color_photo:
        pixel_array = smooth_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixel_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(centers)[labels.flatten()].reshape((h, w, 3))
    else:
        gray_smooth = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2GRAY)
        pixel_array = gray_smooth.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixel_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized_gray = np.uint8(centers)[labels.flatten()].reshape((h, w))
        quantized = cv2.cvtColor(quantized_gray, cv2.COLOR_GRAY2RGB)
    gray_smooth_edge = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_smooth_edge, threshold1=50, threshold2=120)
    if thickness > 0:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    cartoon = quantized.copy()
    cartoon[edges > 0] = [0, 0, 0]
    return cartoon


def cartoon_sketch(img_rgb, line_thickness=1, posterize_levels=3,
                   shadow_color=(20,10,5), mid_color=(120,80,40), highlight_color=(255,248,220)):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if line_thickness > 0:
        k = np.ones((line_thickness+1, line_thickness+1), np.uint8)
        binary = cv2.erode(binary, k, iterations=1)
    factor = 255 // (posterize_levels - 1)
    posterized = (binary // factor) * factor
    lut_r = np.zeros(256, dtype=np.uint8)
    lut_g = np.zeros(256, dtype=np.uint8)
    lut_b = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < 128:
            t = i / 127.0
            lut_r[i] = int(shadow_color[0]*(1-t)+mid_color[0]*t)
            lut_g[i] = int(shadow_color[1]*(1-t)+mid_color[1]*t)
            lut_b[i] = int(shadow_color[2]*(1-t)+mid_color[2]*t)
        else:
            t = (i-128)/127.0
            lut_r[i] = int(mid_color[0]*(1-t)+highlight_color[0]*t)
            lut_g[i] = int(mid_color[1]*(1-t)+highlight_color[1]*t)
            lut_b[i] = int(mid_color[2]*(1-t)+highlight_color[2]*t)
    return cv2.merge([cv2.LUT(posterized,lut_r), cv2.LUT(posterized,lut_g), cv2.LUT(posterized,lut_b)])


SKETCH_STYLES = {
    "Sepia":       dict(shadow_color=(20,10,5),   mid_color=(100,60,20),  highlight_color=(255,245,210)),
    "Comic Blue":  dict(shadow_color=(0,0,80),    mid_color=(30,80,200),  highlight_color=(200,225,255)),
    "Forest":      dict(shadow_color=(5,20,5),    mid_color=(30,100,30),  highlight_color=(200,240,190)),
    "Sunset":      dict(shadow_color=(60,0,0),    mid_color=(200,80,0),   highlight_color=(255,240,180)),
    "Classic B&W": dict(shadow_color=(0,0,0),     mid_color=(128,128,128),highlight_color=(255,255,255)),
}


def img_to_base64(img_rgb):
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


stored_image = {}

# Load HTML from same folder as this script (works on Windows, Mac, Linux)
HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html'), encoding='utf-8').read()

@app.route('/')
def index():
    return HTML

@app.route('/upload', methods=['POST'])
def upload():
    try:
        print(">>> /upload called")
        print(">>> Files received:", list(request.files.keys()))

        if 'image' not in request.files:
            print(">>> ERROR: no 'image' key in request.files")
            return jsonify({'error': 'No image field in request'}), 400

        file = request.files['image']
        print(f">>> File: {file.filename}, type: {file.content_type}")

        img = np.array(Image.open(file.stream).convert('RGB'))
        print(f">>> Image shape: {img.shape}")

        stored_image['data'] = img
        preview = img_to_base64(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        is_color = bool(hsv[:,:,1].mean() > 15)

        print(">>> Upload successful!")
        return jsonify({'preview': preview, 'is_color': is_color})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'data' not in stored_image:
            return jsonify({'error': 'No image uploaded'}), 400
        img = stored_image['data']
        data = request.json
        mode = data.get('mode', 'heavy')
        if mode == 'heavy':
            result = apply_heavy_cartoon(
                img,
                k=int(data.get('k', 8)),
                smoothing_passes=int(data.get('smoothing', 10)),
                thickness=int(data.get('thickness', 1))
            )
        else:
            style = data.get('style', 'Sepia')
            kwargs = SKETCH_STYLES.get(style, SKETCH_STYLES['Sepia'])
            result = cartoon_sketch(
                img,
                line_thickness=int(data.get('line_thickness', 1)),
                posterize_levels=int(data.get('posterize_levels', 3)),
                **kwargs
            )
        return jsonify({'result': img_to_base64(result)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    try:
        if 'data' not in stored_image:
            return jsonify({'error': 'No image'}), 400
        img = stored_image['data']
        data = request.json
        mode = data.get('mode', 'heavy')
        if mode == 'heavy':
            result = apply_heavy_cartoon(img,
                k=int(data.get('k', 8)),
                smoothing_passes=int(data.get('smoothing', 10)),
                thickness=int(data.get('thickness', 1)))
        else:
            style = data.get('style', 'Sepia')
            kwargs = SKETCH_STYLES.get(style, SKETCH_STYLES['Sepia'])
            result = cartoon_sketch(img,
                line_thickness=int(data.get('line_thickness', 1)),
                posterize_levels=int(data.get('posterize_levels', 3)), **kwargs)
        pil = Image.fromarray(result)
        buf = io.BytesIO()
        pil.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png', as_attachment=True, download_name='cartoon.png')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050)
