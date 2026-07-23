import os
import json
import subprocess
import urllib.request
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit

# Manually load .env file if it exists to avoid python-dotenv dependency
env_file_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file_path):
    with open(env_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().strip('"').strip("'")
                    os.environ[key] = val

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hologram-speaker-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory song playlist
SONGS = [
    {
        "id": 1,
        "title": "소문의 낙원",
        "artist": "AKMU",
        "src": "/static/assets/somun.mp3",
        "cover": "/static/assets/somun_art.jpg"
    },
    {
        "id": 2,
        "title": "Lips Hips Kiss",
        "artist": "KISS OF LIFE",
        "src": "/static/assets/lipshipskiss.mp3",
        "cover": "/static/assets/lipshipskiss_art.jpg"
        
    },
    {
        "id": 3,
        "title": "Antifreeze",
        "artist": "Yerin Baek",
        "src": "/static/assets/antifreeze.mp3",
        "cover": "/static/assets/antifreeze_art.jpg"
    },
    {
        "id": 4,
        "title": "Billie Jean",
        "artist": "Michael Jackson",
        "src": "/static/assets/billiejean.mp3",
        "cover": "/static/assets/billiejean_art.jpeg"
    },
    {
        "id": 5,
        "title": "챠우챠우",
        "artist": "Deli Spice",
        "src": "/static/assets/chau_chau.mp3",
        "cover": "/static/assets/chau_chau_art.jpg"
    },
    {
        "id": 6,
        "title": "Kiss Me More",
        "artist": "Doja Cat",
        "src": "/static/assets/kissmemore.mp3",
        "cover": "/static/assets/kissmemore_art.jpg"
    },
    {
        "id": 7,
        "title": "사랑했나봐",
        "artist": "윤도현",
        "src": "/static/assets/musthaveloved.mp3",
        "cover": "/static/assets/musthaveloved_art.webp"
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/songs', methods=['GET'])
def get_songs():
    return jsonify(SONGS)

def create_status_frame(message):
    """Generate a visual status banner image when camera frame is pending"""
    try:
        import numpy as np
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw status text
        cv2.putText(img, "Hologram Speaker Camera Stream", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, message, (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    except Exception:
        return b''

# Hybrid MediaPipe AI & OpenCV Motion Gesture Engine
mp_hands_detector = None
mp_drawing = None
mp_hands_solution = None

try:
    import mediapipe as mp
    mp_hands_solution = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands_detector = mp_hands_solution.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("[Camera Stream] MediaPipe AI 21-Keypoint Hand Engine loaded successfully into app.py!", flush=True)
except Exception as e:
    mp_hands_detector = None
    print(f"[Camera Stream] Operating in Pure OpenCV Motion Detection Mode ({e})", flush=True)

mp_history = []
mp_last_gesture_time = 0
is_pinching = False
pinch_start_time = 0
mp_last_seek_time = 0
mp_prev_point_x = None
mp_prev_point_y = None

prev_motion_gray = None
motion_history = []
last_motion_gesture_time = 0

def process_motion_gesture(frame):
    global mp_hands_detector, mp_drawing, mp_hands_solution, mp_history, mp_last_gesture_time
    global is_pinching, pinch_start_time, mp_last_seek_time
    global mp_prev_point_x, mp_prev_point_y
    global prev_motion_gray, motion_history, last_motion_gesture_time
    try:
        import cv2
        import numpy as np
        import time

        current_time = time.time()

        # 1. Preferred AI Mode: MediaPipe 21-Keypoint Tracker (99.9% Accuracy)
        if mp_hands_detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands_detector.process(rgb_frame)

            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    index_pip = hand_landmarks.landmark[6]
                    middle_tip = hand_landmarks.landmark[12]
                    middle_pip = hand_landmarks.landmark[10]
                    middle_mcp = hand_landmarks.landmark[9]
                    ring_tip = hand_landmarks.landmark[16]
                    ring_pip = hand_landmarks.landmark[14]
                    wrist = hand_landmarks.landmark[0]

                    # 1. Calculate Hand Physical Size (Wrist to Middle MCP)
                    hand_size = np.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)

                    # --- PROXIMITY GUARD: Ignore hands that are very far away (< 0.08 of frame) ---
                    if hand_size < 0.08:
                        is_pinching = False
                        mp_prev_point_x = None
                        mp_prev_point_y = None
                        mp_history = []
                        return

                    # 2. Calculate Scale-Invariant Pinch Ratio
                    dist_pinch = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                    pinch_ratio = dist_pinch / max(0.01, hand_size)

                    # Multi-joint Finger Posture Analysis
                    is_index_extended = index_tip.y < index_pip.y
                    is_middle_extended = middle_tip.y < middle_pip.y
                    is_ring_extended = ring_tip.y < ring_pip.y
                    is_open_palm = is_index_extended and is_middle_extended and is_ring_extended
                    is_index_only = is_index_extended and (not is_middle_extended) and (not is_ring_extended)

                    # --- BOUNDARY GUARD: Suppress swipe when hand is near frame edges ---
                    is_near_edge = (wrist.x < 0.12 or wrist.x > 0.88 or wrist.y < 0.10 or wrist.y > 0.90)

                    # --- 1. PINCH & PINCH-DRAG SEEK (Scale-Invariant Hysteresis) ---
                    if (pinch_ratio < 0.28 or (is_pinching and pinch_ratio < 0.40)) and not is_open_palm and not is_near_edge:
                        mp_history = []
                        mp_prev_point_x = None
                        mp_prev_point_y = None

                        if not is_pinching:
                            is_pinching = True
                            pinch_start_time = current_time

                        # Pinch held > 0.20s -> Enter PINCH DRAG SEEK MODE!
                        if current_time - pinch_start_time > 0.20:
                            if current_time - mp_last_seek_time > 0.08:
                                progress_ratio = 1.0 - index_tip.x
                                progress_ratio = max(0.0, min(1.0, progress_ratio))
                                print(f"[Camera Gesture - MediaPipe AI] Scale-Invariant Pinch Drag Seeking to {int(progress_ratio*100)}%", flush=True)
                                socketio.emit('gesture_trigger', {'type': 'seek', 'value': progress_ratio})
                                mp_last_seek_time = current_time
                        return
                    else:
                        # Pinch just released
                        if is_pinching:
                            pinch_duration = current_time - pinch_start_time
                            is_pinching = False
                            if pinch_duration < 0.35: # Quick pinch tap -> Play/Pause
                                print("[Camera Gesture - MediaPipe AI] Quick Pinch Tap! -> Play/Pause Toggle", flush=True)
                                socketio.emit('gesture_trigger', {'type': 'play_pause'})
                                mp_last_gesture_time = current_time
                                mp_history = []
                                return

                    # --- 2. INDEX FINGER POINTING (3D Cube Space Touch Rotation) ---
                    if is_index_only and pinch_ratio > 0.35 and not is_near_edge:
                        mp_history = []
                        curr_px = 1.0 - index_tip.x # Mirrored X
                        curr_py = index_tip.y
                        if mp_prev_point_x is not None and mp_prev_point_y is not None:
                            raw_dx = curr_px - mp_prev_point_x
                            raw_dy = curr_py - mp_prev_point_y
                            # Gentle, softened delta values with deadzone to eliminate hyper-sensitivity
                            dx = max(-0.015, min(0.015, raw_dx)) * 0.40
                            dy = max(-0.015, min(0.015, raw_dy)) * 0.40
                            if abs(raw_dx) > 0.003 or abs(raw_dy) > 0.003:
                                print(f"[Camera Gesture - MediaPipe AI] Gentle 3D Cube Rotation (dx: {dx:.3f}, dy: {dy:.3f})", flush=True)
                                socketio.emit('gesture_trigger', {
                                    'type': 'rotate',
                                    'dx': dx,
                                    'dy': dy,
                                    'x': curr_px,
                                    'y': curr_py,
                                    'touch': True
                                })
                        mp_prev_point_x = curr_px
                        mp_prev_point_y = curr_py
                        return
                    else:
                        mp_prev_point_x = None
                        mp_prev_point_y = None

                    # --- 3. OPEN HAND SWIPE DETECTION (Next/Prev Song) ---
                    if is_open_palm and pinch_ratio > 0.35:
                        if is_near_edge:
                            mp_history = []
                            return

                        if current_time - mp_last_gesture_time >= 1.0:
                            mp_history.append((wrist.x, current_time))
                            if len(mp_history) > 6:
                                mp_history.pop(0)

                            if len(mp_history) >= 2:
                                dx_total = mp_history[-1][0] - mp_history[0][0]
                                dt_total = mp_history[-1][1] - mp_history[0][1]

                                if 0.04 <= dt_total <= 0.65:
                                    if dx_total > 0.05: # Swipe Right across camera
                                        print("[Camera Gesture - MediaPipe AI] Open Palm Swipe Right -> Next Track", flush=True)
                                        socketio.emit('gesture_trigger', {'type': 'next'})
                                        mp_last_gesture_time = current_time
                                        mp_history = []
                                        return
                                    elif dx_total < -0.05: # Swipe Left across camera
                                        print("[Camera Gesture - MediaPipe AI] Open Palm Swipe Left -> Previous Track", flush=True)
                                        socketio.emit('gesture_trigger', {'type': 'prev'})
                                        mp_last_gesture_time = current_time
                                        mp_history = []
                                        return
                        return
            else:
                is_pinching = False
                mp_history = []
                mp_prev_point_x = None
                mp_prev_point_y = None
            return

        # 2. Fallback Mode: OpenCV Motion Trajectory Detector
        if current_time - last_motion_gesture_time < 0.9:
            motion_history = []
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_motion_gray is None or prev_motion_gray.shape != gray.shape:
            prev_motion_gray = gray
            return

        frame_diff = cv2.absdiff(prev_motion_gray, gray)
        thresh = cv2.threshold(frame_diff, 28, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        prev_motion_gray = gray

        total_diff_pixels = np.count_nonzero(thresh)
        total_screen_pixels = frame.shape[0] * frame.shape[1]
        if (total_diff_pixels / float(total_screen_pixels)) > 0.40:
            motion_history = []
            return

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 2000]

        if valid_contours:
            largest_c = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_c)
            M = cv2.moments(largest_c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                motion_history.append((cx, cy, area, current_time))
                if len(motion_history) > 6:
                    motion_history.pop(0)

                if len(motion_history) >= 3:
                    dx_total = motion_history[-1][0] - motion_history[0][0]
                    area_start = motion_history[0][2]
                    area_end = motion_history[-1][2]
                    dt_total = motion_history[-1][3] - motion_history[0][3]

                    if 0.08 <= dt_total <= 0.55:
                        if abs(dx_total) < 35 and (area_start - area_end) > 1000:
                            print("[Camera Gesture] Pinch / Tap detected! -> Play/Pause Toggle", flush=True)
                            socketio.emit('gesture_trigger', {'type': 'play_pause'})
                            last_motion_gesture_time = current_time
                            motion_history = []
                            return

                        if dx_total > 75:
                            print("[Camera Gesture] Clean Swipe Right detected! -> Next Track", flush=True)
                            socketio.emit('gesture_trigger', {'type': 'next'})
                            last_motion_gesture_time = current_time
                            motion_history = []
                        elif dx_total < -75:
                            print("[Camera Gesture] Clean Swipe Left detected! -> Previous Track", flush=True)
                            socketio.emit('gesture_trigger', {'type': 'prev'})
                            last_motion_gesture_time = current_time
                            motion_history = []
        else:
            motion_history = []
    except Exception:
        pass

def gen_camera_frames():
    """MJPEG Live Camera Streamer with built-in Gesture Detection for Pi 5"""
    import subprocess
    import shutil
    import cv2
    import numpy as np

    # 1. Try native Raspberry Pi 5 libcamera streamer (rpicam-vid / libcamera-vid)
    cam_cmd = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
    if cam_cmd:
        proc = None
        try:
            # Kill any previous background camera process so device isn't busy
            subprocess.run(["pkill", "-9", "-f", "rpicam-vid"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run(["pkill", "-9", "-f", "libcamera-vid"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            print(f"[Camera Stream] Starting Pi 5 libcamera process: {cam_cmd}")
            proc = subprocess.Popen(
                [cam_cmd, "-t", "0", "--inline", "--width", "640", "--height", "480", "--codec", "mjpeg", "--framerate", "30", "-o", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            buffer = b""
            frame_counter = 0
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                a = buffer.find(b'\xff\xd8')
                b = buffer.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = buffer[a:b+2]
                    buffer = buffer[b+2:]
                    
                    frame_counter += 1
                    # Analyze MediaPipe AI motion gesture on alternating frames (15 FPS AI, 30 FPS video)
                    if frame_counter % 2 == 0:
                        frame_decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame_decoded is not None:
                            process_motion_gesture(frame_decoded)

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        except Exception as e:
            print(f"[Camera Stream rpicam-vid Error] {e}")
        finally:
            if proc:
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            return

    # 2. Fallback to OpenCV V4L2 device probing
    try:
        import cv2
        cap = None
        for idx in [20, 19, 21, 0, 1]:
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if hasattr(cv2, 'CAP_V4L2') else cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    print(f"[Camera Stream] Successfully opened camera device index: {idx}")
                    break
                temp_cap.release()
        
        if not cap or not cap.isOpened():
            print("[Camera Stream] Cannot open camera device. Displaying status banner.")
            status_bytes = create_status_frame("Camera Hardware Not Found / Disconnected")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + status_bytes + b'\r\n')
            return

        while True:
            success, frame = cap.read()
            if not success or frame is None:
                status_bytes = create_status_frame("Waiting for Camera Frame...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + status_bytes + b'\r\n')
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    except Exception as e:
        print(f"[Camera Stream Error] {e}")

@app.route('/video_feed')
def video_feed():
    """Live Camera MJPEG Video Stream Route"""
    return Response(gen_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/log', methods=['POST'])
def client_log():
    data = request.json
    print(f"\n[CLIENT ERROR] {data.get('message')}\nSource: {data.get('source')} (Line {data.get('lineno')}, Col {data.get('colno')})\nStack: {data.get('error')}\n", flush=True)
    return jsonify(status="ok")

@app.route('/api/gemini', methods=['POST'])
def query_gemini():
    data = request.json or {}
    prompt = data.get('prompt', '')
    
    api_key = os.environ.get("GEMINI_API_KEY")
    answer = None
    
    if not api_key:
        print("[Gemini] GEMINI_API_KEY not found in environment. Falling back to Mock responses.")
        mock_responses = {
            "안녕": "안녕하세요! 저는 제미나이 홀로그램 비서입니다. 무엇이든 물어보세요.",
            "날씨": "오늘 홀로그램 룸의 날씨는 미래지향적인 네온 블루 아우라로 가득하며, 매우 쾌적합니다.",
            "누구": "저는 라즈베리파이 5와 제미나이 2.5 Flash API로 작동하는 차세대 홀로그램 음악 스피커의 인공지능 비서입니다.",
            "음악": "음악은 감정을 움직이는 가장 훌륭한 형태의 마법입니다. 원하시는 곡을 핀치 제스처나 버튼으로 조작해 보세요.",
            "노래": "음악은 감정을 움직이는 가장 훌륭한 형태의 마법입니다. 원하시는 곡을 핀치 제스처나 버튼으로 조작해 보세요.",
            "재생": "[ACTION:PLAY] 음악을 재생할게요.",
            "틀어줘": "[ACTION:PLAY] 음악을 틀어드릴게요.",
            "멈춰": "[ACTION:PAUSE] 음악을 일시정지할게요.",
            "정지": "[ACTION:PAUSE] 음악을 멈출게요.",
            "다음": "[ACTION:NEXT] 다음 곡을 재생할게요.",
            "이전": "[ACTION:PREV] 이전 곡을 재생할게요.",
        }
        answer = "죄송해요, API 키가 설정되지 않은 모의 모드입니다. 다른 재미있는 대화를 시도해 보시거나, 환경 변수에 GEMINI_API_KEY를 추가해 주세요!"
        for key, val in mock_responses.items():
            if key in prompt:
                answer = val
                break
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "systemInstruction": {
                "parts": [{
                    "text": "너는 스마트 스피커 홀로그램 비서야. 사용자가 음악 재생, 일시 정지(멈춤), 다음 곡, 이전 곡 등을 요구하면 답변의 처음에 반드시 알맞은 태그를 달아줘:\n"
                            "- 음악 재생 요구 시: [ACTION:PLAY]\n"
                            "- 음악 일시정지/정지 요구 시: [ACTION:PAUSE]\n"
                            "- 다음 곡 요구 시: [ACTION:NEXT]\n"
                            "- 이전 곡 요구 시: [ACTION:PREV]\n"
                            "예: [ACTION:PLAY] 음악을 재생할게요.\n"
                            "만약 일반적인 질문이나 대화라면 태그를 달지 마."
                }]
            },
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 512,
                "thinkingConfig": {
                    "thinkingBudget": 0
                }
            }
        }
        
        try:
            print(f"[Gemini Request] Prompt: {prompt}", flush=True)
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                answer = res_data['candidates'][0]['content']['parts'][0]['text']
                print(f"[Gemini Response] Answer: {answer}", flush=True)
        except Exception as e:
            print(f"[Gemini Error] API call failed: {e}", flush=True)
            answer = f"제미나이 API 호출 중 오류가 발생했습니다: {str(e)}"

    # Parse control tags from the final answer
    action_type = None
    if answer:
        if "[ACTION:PLAY]" in answer:
            action_type = "play"
            answer = answer.replace("[ACTION:PLAY]", "").strip()
        elif "[ACTION:PAUSE]" in answer:
            action_type = "pause"
            answer = answer.replace("[ACTION:PAUSE]", "").strip()
        elif "[ACTION:NEXT]" in answer:
            action_type = "next"
            answer = answer.replace("[ACTION:NEXT]", "").strip()
        elif "[ACTION:PREV]" in answer:
            action_type = "prev"
            answer = answer.replace("[ACTION:PREV]", "").strip()

    if action_type:
        print(f"[Gemini Action] Detected action: {action_type}. Emitting socket event.", flush=True)
        socketio.emit('gesture_trigger', {'type': action_type})

    return jsonify(response=answer)

@app.route('/control')
def remote_control_page():
    return render_template('control.html')

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# WebSocket events for gesture forwarding (e.g. from OpenCV Python script or IR sensor)
@socketio.on('connect')
def handle_connect():
    print('[WebSocket] Client connected')

@socketio.on('hardware_gesture')
def handle_hardware_gesture(data):
    """
    Called by an external python script running gesture recognition (like MediaPipe or APDS-9960).
    Forwards gesture event to frontend.
    data format: { 'type': 'swipe_left'/'swipe_right'/'push'/'volume_up'/'volume_down', 'value': velocity_float }
    """
    print(f"[WebSocket] Received hardware gesture: {data}")
    # Forward the gesture straight to the browser
    emit('gesture_trigger', data, broadcast=True)

# --- USB Microphone Gemini Audio Capture Routine ---
def find_usb_mic_device():
    try:
        res = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        out = res.stdout
        for line in out.splitlines():
            if "card" in line and ("USB" in line or "Codec" in line or "PnP" in line):
                card_num = line.split("card")[1].split(":")[0].strip()
                print(f"[USB Mic Auto-Detect] Found USB Audio Card {card_num}", flush=True)
                return [f"plughw:{card_num},0", "default", "pulse", f"hw:{card_num},0"]
    except Exception as e:
        print(f"[USB Mic Find Error] {e}", flush=True)
    return ["default", "plughw:2,0", "pulse", "hw:2,0"]

def record_and_process_audio():
    import base64
    mic_dev_candidates = find_usb_mic_device()
    wav_path = "/tmp/gemini_input.wav"
    captured = False
    used_device = None

    for mic_dev in mic_dev_candidates:
        print(f"[Gemini Mic] Trying audio capture with device '{mic_dev}'...", flush=True)
        try:
            res = subprocess.run(["arecord", "-D", mic_dev, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "4", wav_path], capture_output=True, text=True, timeout=6)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 2000:
                captured = True
                used_device = mic_dev
                print(f"[Gemini Mic] Capture successful with '{mic_dev}'! File size: {os.path.getsize(wav_path)} bytes.", flush=True)
                break
            else:
                print(f"[Gemini Mic] Device '{mic_dev}' failed or empty. Stderr: {res.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"[Gemini Mic] Error with device '{mic_dev}': {e}", flush=True)

    if not captured:
        print("[Gemini Mic Error] All recording devices failed!", flush=True)
        socketio.emit('gemini_mic_response', {'response': "마이크 녹음에 실패했습니다. alsamixer에서 USB 마이크가 켜져 있는지 확인해 주세요."})
        return

    socketio.emit('gemini_mic_state', {'state': 'THINKING'})

    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            answer = "USB 마이크 음성을 정상 수신했습니다! 환경 변수에 GEMINI_API_KEY를 추가하시면 실시간 대화와 음성 곡 제어가 작동합니다."
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "systemInstruction": {
                    "parts": [{
                        "text": "너는 라즈베리파이 홀로그램 스마트 스피커 AI 비서야. 사용자가 한국어 음성으로 요구한 내용을 분석해 친절히 답변해줘. 만약 음악 재생, 일시정지, 다음 곡, 이전 곡 등의 요구가 있다면 답변 머리에 알맞은 태그([ACTION:PLAY], [ACTION:PAUSE], [ACTION:NEXT], [ACTION:PREV])를 반드시 달아줘."
                    }]
                },
                "contents": [{
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": audio_b64
                            }
                        },
                        {
                            "text": "사용자의 마이크 음성 명령어입니다. 음성을 인식하고 적절한 답변과 제어 태그를 반환하세요."
                        }
                    ]
                }],
                "generationConfig": {
                    "maxOutputTokens": 512,
                    "thinkingConfig": { "thinkingBudget": 0 }
                }
            }
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                answer = res_data['candidates'][0]['content']['parts'][0]['text']

        # Parse control tags from the final answer
        action_type = None
        if answer:
            if "[ACTION:PLAY]" in answer:
                action_type = "play"
                answer = answer.replace("[ACTION:PLAY]", "").strip()
            elif "[ACTION:PAUSE]" in answer:
                action_type = "pause"
                answer = answer.replace("[ACTION:PAUSE]", "").strip()
            elif "[ACTION:NEXT]" in answer:
                action_type = "next"
                answer = answer.replace("[ACTION:NEXT]", "").strip()
            elif "[ACTION:PREV]" in answer:
                action_type = "prev"
                answer = answer.replace("[ACTION:PREV]", "").strip()

        if action_type:
            print(f"[Gemini Action] Detected action: {action_type}. Emitting socket event.", flush=True)
            socketio.emit('gesture_trigger', {'type': action_type})

        print(f"[Gemini Mic Answer] {answer}", flush=True)
        socketio.emit('gemini_mic_response', {'response': answer})
    except Exception as e:
        print(f"[Gemini Mic Error] {e}", flush=True)
        socketio.emit('gemini_mic_response', {'response': f"마이크 처리 중 오류가 발생했습니다: {e}"})

@app.route('/api/debug_mic')
def debug_mic():
    import struct
    results = {}
    try:
        r1 = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        results['arecord_l'] = r1.stdout
    except Exception as e:
        results['arecord_l_err'] = str(e)

    wav_path = "/tmp/debug_mic.wav"
    capture_success = False
    used_dev = None

    for dev in ["default", "pulse", "plughw:2,0", "hw:2,0"]:
        try:
            r2 = subprocess.run(["arecord", "-D", dev, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "2", wav_path], capture_output=True, text=True, timeout=5)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                capture_success = True
                used_dev = dev
                results['capture_log'] = f"Success using device '{dev}', file size: {os.path.getsize(wav_path)} bytes"
                break
            else:
                results[f'capture_{dev}_err'] = f"returncode: {r2.returncode}, stderr: {r2.stderr.strip()}"
        except Exception as e:
            results[f'capture_{dev}_err'] = str(e)

    if capture_success and os.path.exists(wav_path):
        try:
            with open(wav_path, "rb") as f:
                raw_data = f.read()[44:]
            samples = struct.unpack(f"<{len(raw_data)//2}h", raw_data)
            rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5 if samples else 0
            results['audio_rms_volume'] = f"{rms:.2f} (RMS energy: >50 means voice recorded, <10 means mute/silence)"
        except Exception as e:
            results['rms_calc_err'] = str(e)

    results['gemini_api_key_present'] = bool(os.environ.get("GEMINI_API_KEY"))
    return jsonify(results)

@app.route('/api/tts')
def tts_proxy():
    text = request.args.get('text', '')
    if not text:
        return "No text provided", 400

    import re
    import urllib.parse
    text = re.sub(r'\[ACTION:[^\]]+\]', '', text).strip()
    if not text:
        return "Empty text", 400

    raw_sentences = re.split(r'([.?!,\n])', text)
    chunks = []
    current = ""
    for s in raw_sentences:
        if len(current) + len(s) < 150:
            current += s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())

    if not chunks:
        chunks = [text[:150]]

    audio_bytes = bytearray()
    for chunk in chunks[:4]:
        tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={urllib.parse.quote(chunk)}&tl=ko&client=tw-ob"
        try:
            req = urllib.request.Request(tts_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=5) as response:
                audio_bytes.extend(response.read())
        except Exception as e:
            print(f"[TTS Proxy Error] {e}", flush=True)

    if not audio_bytes:
        return jsonify({"error": "Failed to generate TTS"}), 500

    return Response(bytes(audio_bytes), mimetype="audio/mpeg")

@socketio.on('gemini_button')
def handle_gemini_button():
    import threading
    print("[WebSocket] Received hardware gemini button event. Starting USB Mic capture...", flush=True)
    emit('gemini_toggle', {}, broadcast=True)
    threading.Thread(target=record_and_process_audio, daemon=True).start()

# Remote control sync events
@socketio.on('remote_setting_change')
def handle_remote_setting_change(data):
    print(f"[WebSocket] Received remote setting update: {data}")
    emit('remote_setting_trigger', data, broadcast=True)

@socketio.on('player_state_sync')
def handle_player_state_sync(data):
    # Broadcast player stats (playing track, progress, mode) to the remote control client
    emit('player_state_update', data, broadcast=True)

@socketio.on('remote_chat')
def handle_remote_chat(data):
    print(f"[WebSocket] Received remote chat query: {data}")
    emit('remote_chat_trigger', data, broadcast=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Hologram Speaker Server on http://0.0.0.0:{port} ...")
    socketio.run(app, debug=True, host='::', port=port)
