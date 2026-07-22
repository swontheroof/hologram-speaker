import os
import json
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

# Integrated Motion Gesture Detector for Pi Camera
prev_motion_gray = None
last_motion_gesture_time = 0

def process_motion_gesture(frame):
    global prev_motion_gray, last_motion_gesture_time
    try:
        import cv2
        import numpy as np
        import time

        current_time = time.time()
        if current_time - last_motion_gesture_time < 0.9: # 0.9s gesture cooldown
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_motion_gray is None or prev_motion_gray.shape != gray.shape:
            prev_motion_gray = gray
            return

        frame_diff = cv2.absdiff(prev_motion_gray, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prev_motion_gray = gray

        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 3000)
        if motion_area > 7000:
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                w = frame.shape[1]
                if cx > int(w * 0.65):
                    print("[Camera Gesture] Motion Swipe Right detected! -> Next Track", flush=True)
                    socketio.emit('hardware_gesture', {'type': 'next'})
                    last_motion_gesture_time = current_time
                elif cx < int(w * 0.35):
                    print("[Camera Gesture] Motion Swipe Left detected! -> Previous Track", flush=True)
                    socketio.emit('hardware_gesture', {'type': 'prev'})
                    last_motion_gesture_time = current_time
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
                    
                    # Analyze motion gesture in real-time
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

@socketio.on('gemini_button')
def handle_gemini_button():
    print("[WebSocket] Received hardware gemini button event")
    # Broadcast to the browser frontend to toggle Gemini Mode
    emit('gemini_toggle', {}, broadcast=True)

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
