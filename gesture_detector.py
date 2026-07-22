import os
import time
import socketio
import sys

# Initialize Socket.IO Client to send events to our Flask backend
sio = socketio.Client()

SERVER_URL = "http://localhost:5001"

@sio.event
def connect():
    print(f"[HW Detector] Connected to Hologram Speaker Server at {SERVER_URL}")

@sio.event
def disconnect():
    print("[HW Detector] Disconnected from server")

def run_mock_keyboard_detector():
    """
    Simulates hardware sensors using terminal keyboard inputs.
    Use this to test the integration on your Mac without a physical camera/sensor.
    """
    print("\n" + "="*50)
    print("  [MOCK SENSOR MODE] Terminal Keyboard Controller")
    print("  Press keys to simulate Raspberry Pi sensor inputs:")
    print("   - [L] / [R] : Swipe Left / Right (Next/Prev track)")
    print("   - [S]       : Fast Swipe (Spin hologram)")
    print("   - [Space]   : Double tap / Pinch (Play/Pause)")
    print("   - [U] / [D] : Volume Up / Down")
    print("   - [B]       : Gemini Assistant Button (Toggle voice mode)")
    print("   - [Q]       : Exit simulator")
    print("="*50 + "\n")

    try:
        import tty
        import termios
    except ImportError:
        # Fallback for Windows/Non-unix environment if any
        print("Terminal control is designed for macOS/Linux shells.")
        return

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    while True:
        char = getch()
        if char.lower() == 'q':
            print("\nExiting sensor simulation...")
            break
        elif char.lower() == 'l':
            print("\n[Sensor] Swipe Left -> Previous Track")
            sio.emit('hardware_gesture', {'type': 'prev'})
        elif char.lower() == 'r':
            print("\n[Sensor] Swipe Right -> Next Track")
            sio.emit('hardware_gesture', {'type': 'next'})
        elif char == ' ':
            print("\n[Sensor] Pinch Gesture -> Play/Pause Toggle")
            sio.emit('hardware_gesture', {'type': 'play_pause'})
        elif char.lower() == 's':
            print("\n[Sensor] High-speed Swipe -> Spinning Hologram")
            # Send a fast spin velocity (e.g. 0.15 rad/s)
            sio.emit('hardware_gesture', {'type': 'swipe', 'value': 0.12})
        elif char.lower() == 'u':
            print("\n[Sensor] Volume Up -> Seek Forward")
            sio.emit('hardware_gesture', {'type': 'seek', 'value': 0.8}) # mock value
        elif char.lower() == 'd':
            print("\n[Sensor] Volume Down -> Seek Backward")
            sio.emit('hardware_gesture', {'type': 'seek', 'value': 0.2})
        elif char.lower() == 'b':
            print("\n[Sensor] Gemini Button -> Toggle Voice Assistant Mode")
            sio.emit('gemini_button', {})

def run_physical_camera_detector():
    """
    Boilerplate code for Raspberry Pi Camera + OpenCV + MediaPipe Hands.
    To run this on your Raspberry Pi:
      1. pip install opencv-python mediapipe python-socketio
      2. Run: python3 gesture_detector.py --camera
    """
    print("[HW Detector] Camera mode starting... Importing OpenCV and MediaPipe...")
    has_mediapipe = False
    try:
        import cv2
        try:
            import mediapipe as mp
            has_mediapipe = True
        except ImportError:
            print("\n[HW Detector] MediaPipe not found. Operating in Pure OpenCV Motion Detection mode!")
    except ImportError:
        print("\n[ERROR] OpenCV is not installed. Please run: pip install opencv-python")
        return

    hands = None
    if has_mediapipe:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    import subprocess
    import shutil
    import numpy as np

    # Kill any previous camera processes
    subprocess.run(["pkill", "-9", "-f", "rpicam-vid"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "libcamera-vid"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    cam_cmd = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
    use_rpicam = False
    proc = None

    if cam_cmd:
        try:
            print(f"[Camera Detector] Starting Pi 5 libcamera stream for gesture detection: {cam_cmd}")
            proc = subprocess.Popen(
                [cam_cmd, "-t", "0", "--inline", "--width", "640", "--height", "480", "--codec", "mjpeg", "--framerate", "30", "-o", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            use_rpicam = True
        except Exception as e:
            print(f"[Camera Detector rpicam-vid Error] {e}")

    cap = None
    if not use_rpicam:
        for idx in [20, 0, 19, 1]:
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if hasattr(cv2, 'CAP_V4L2') else cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    print(f"[Camera Detector] Successfully opened camera device index: {idx}")
                    break
                temp_cap.release()

    if not use_rpicam and (not cap or not cap.isOpened()):
        print("[ERROR] Cannot open webcam camera on any index.")
        return

    print("\n" + "="*50)
    print("  [CAMERA GESTURE MODE] Active")
    print("  Show your hand to the camera:")
    print("   - Swipe Left/Right  : Prev/Next track")
    print("   - Pinch Index/Thumb : Play/Pause toggle")
    print("   - Press 'q' in the window to quit")
    print("="*50 + "\n")

    prev_x = None
    prev_y = None
    last_swipe_time = 0
    last_pinch_time = 0
    is_pinch_active = False
    last_pose = None
    current_interaction_lock = None
    lock_cooldown_until = 0
    interaction_mode = 'rotate'
    last_vx = 0
    last_vy = 0
    is_dragging = False
    tracking_loss_frames = 0
    pose_loss_frames = 0

    buffer = b""
    try:
        while True:
            frame = None
            if use_rpicam and proc:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                a = buffer.find(b'\xff\xd8')
                b = buffer.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = buffer[a:b+2]
                    buffer = buffer[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                else:
                    continue
            elif cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            else:
                break

            # Flip the image horizontally for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            current_time = time.time()
            results = None
            if has_mediapipe:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

            if results and results.multi_hand_landmarks:
                tracking_loss_frames = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    # Index tip (8), Thumb tip (4), Wrist (0)
                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    wrist = hand_landmarks.landmark[0]

                    # Helper to calculate 3D distance
                    def get_dist(p1, p2):
                        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5

                    # Check finger extension status (Tip is further from Wrist than PIP joint)
                    is_index_extended = get_dist(wrist, hand_landmarks.landmark[8]) > get_dist(wrist, hand_landmarks.landmark[6])
                    is_middle_extended = get_dist(wrist, hand_landmarks.landmark[12]) > get_dist(wrist, hand_landmarks.landmark[10])
                    is_ring_extended = get_dist(wrist, hand_landmarks.landmark[16]) > get_dist(wrist, hand_landmarks.landmark[14])
                    is_pinky_extended = get_dist(wrist, hand_landmarks.landmark[20]) > get_dist(wrist, hand_landmarks.landmark[18])

                    is_v_sign = is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended
                    is_pointing = is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended
                    is_fist = not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended
                    is_flat_palm = is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended

                    # Define Edge Zone check (8% border margin)
                    edge_margin = 0.08
                    is_at_edge = (index_tip.x < edge_margin or index_tip.x > (1.0 - edge_margin) or index_tip.y < edge_margin or index_tip.y > (1.0 - edge_margin))

                    # Mode Transition State Machine (ONLY if hand is NOT at the edge or we already have a lock)
                    if not is_at_edge or current_interaction_lock is not None:
                        if is_pointing:
                            if interaction_mode != 'rotate':
                                interaction_mode = 'rotate'
                                prev_x = None
                                prev_y = None
                                is_pinch_active = False
                        elif is_flat_palm:
                            if interaction_mode != 'music':
                                interaction_mode = 'music'
                                prev_x = None
                                prev_y = None
                                is_dragging = False

                    # 1. Calculate hand scale (Z-depth approximation)
                    index_knuckle = hand_landmarks.landmark[5]
                    hand_scale = ((wrist.x - index_knuckle.x)**2 + (wrist.y - index_knuckle.y)**2)**0.5
                    is_touch_active = hand_scale >= 0.14

                    # Track pose state changes to prevent coordinate jumps
                    current_pose = 'pointing' if is_pointing else ('v' if is_v_sign else ('palm' if is_flat_palm else None))
                    if current_pose != last_pose:
                        prev_x = None
                        prev_y = None
                        last_pose = current_pose

                    # Calculate pinch distance early
                    dist = ((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)**0.5
                    is_currently_pinching = dist < 0.08 and not is_fist

                    # --- Mutex State Machine Lock with Cooldown ---
                    lock_released = False

                    if current_interaction_lock == 'pinch':
                        if not is_currently_pinching and not is_pinch_active:
                            current_interaction_lock = None
                            lock_released = True
                    elif current_interaction_lock == 'swipe':
                        if not is_v_sign:
                            current_interaction_lock = None
                            lock_released = True
                    elif current_interaction_lock == 'rotate':
                        is_pose_still_valid = is_touch_active and is_pointing and interaction_mode == 'rotate'
                        if is_pose_still_valid:
                            pose_loss_frames = 0
                        else:
                            if interaction_mode != 'rotate':
                                # Explicit palm mode transition
                                current_interaction_lock = None
                                lock_released = True
                                pose_loss_frames = 0
                            else:
                                # Pose misclassification buffer
                                pose_loss_frames += 1
                                if pose_loss_frames > 6:
                                    current_interaction_lock = None
                                    lock_released = True
                                    pose_loss_frames = 0

                    # If lock was released, trigger a 650ms cooldown
                    if lock_released:
                        lock_cooldown_until = current_time + 0.65

                    # Assign new lock ONLY if not in cooldown AND hand is not at the edge (ignores edge entries)
                    if current_interaction_lock is None and current_time >= lock_cooldown_until and not is_at_edge:
                        if is_currently_pinching and interaction_mode == 'music':
                            current_interaction_lock = 'pinch'
                        elif is_v_sign and interaction_mode == 'music':
                            current_interaction_lock = 'swipe'
                        elif is_touch_active and is_pointing and interaction_mode == 'rotate':
                            current_interaction_lock = 'rotate'
                            # Reset baselines for rotation tracking
                            prev_x = index_tip.x
                            prev_y = index_tip.y

                    # Determine allowed modes
                    allow_swipe = (current_interaction_lock == 'swipe' or current_interaction_lock is None) and current_time >= lock_cooldown_until and interaction_mode == 'music'
                    allow_rotate = (current_interaction_lock == 'rotate' or current_interaction_lock is None) and current_time >= lock_cooldown_until and interaction_mode == 'rotate'
                    allow_pinch = (current_interaction_lock == 'pinch' or current_interaction_lock is None) and current_time >= lock_cooldown_until and interaction_mode == 'music'

                    # If hand is at edge and not interacting, block interaction initiation
                    if is_at_edge and current_interaction_lock is None:
                        prev_x = None
                        prev_y = None
                        is_pinch_active = False
                        is_dragging = False
                        continue

                    # Only process active hand shapes in their allowed modes
                    # If we are in the pose loss grace period, we bypass validation failure to prevent aborting drag
                    is_pose_loss_grace_active = (current_interaction_lock == 'rotate' and pose_loss_frames > 0)
                    has_valid_pose = (is_v_sign and allow_swipe) or (is_pointing and allow_rotate) or (is_currently_pinching and allow_pinch) or (is_flat_palm and interaction_mode == 'music') or is_pose_loss_grace_active
                    if not has_valid_pose:
                        prev_x = None
                        prev_y = None
                        is_pinch_active = False
                        if current_interaction_lock is not None:
                            current_interaction_lock = None
                            lock_cooldown_until = current_time + 0.65
                        continue

                    # 1. Detect Swipe Gestures (ONLY when showing V-Sign and allowed)
                    if is_v_sign and allow_swipe:
                        if prev_x is not None:
                            dx = index_tip.x - prev_x
                            # Speed/distance thresholds for V-swipe gesture
                            if abs(dx) > 0.15 and (current_time - last_swipe_time) > 0.8:
                                if dx > 0:
                                    print(f"[Camera Gesture] V-Swipe Right detected! -> Next Track")
                                    sio.emit('hardware_gesture', {'type': 'next'})
                                else:
                                    print(f"[Camera Gesture] V-Swipe Left detected! -> Previous Track")
                                    sio.emit('hardware_gesture', {'type': 'prev'})
                                last_swipe_time = current_time
                        prev_x = index_tip.x
                        prev_y = index_tip.y
                    
                    # 2. Detect Direct Drag Rotation (ONLY when Pointing Finger and allowed)
                    elif (is_pointing or is_pose_loss_grace_active) and allow_rotate:
                        cursor_x = index_tip.x
                        cursor_y = index_tip.y
                        
                        is_dragging = is_touch_active or is_pose_loss_grace_active
                        
                        # Establish baseline tracking on starting drag to prevent jumps!
                        if is_dragging:
                            if prev_x is None or prev_y is None:
                                prev_x = cursor_x
                                prev_y = cursor_y
                                last_vx = 0
                                last_vy = 0
                        
                        # Freeze tracking updates during pose loss grace frames
                        if prev_x is not None and prev_y is not None and not is_pose_loss_grace_active:
                            dx = cursor_x - prev_x
                            dy = cursor_y - prev_y
                            if abs(dx) > 0.001 or abs(dy) > 0.001:
                                if is_dragging:
                                    last_vx = dx / 0.033
                                    last_vy = dy / 0.033
                                sio.emit('hardware_gesture', {
                                    'type': 'rotate',
                                    'dx': dx,
                                    'dy': dy,
                                    'x': cursor_x,
                                    'y': cursor_y,
                                    'grab': False,
                                    'touch': is_touch_active
                                })
                        if not is_pose_loss_grace_active:
                            prev_x = cursor_x
                            prev_y = cursor_y

                    # 3. Detect Pinch/Tap (ONLY when allowed)
                    if is_currently_pinching and allow_pinch:
                        if not is_pinch_active and (current_time - last_pinch_time) > 1.2:
                            print(f"[Camera Gesture] Pinch detected! -> Play/Pause")
                            sio.emit('hardware_gesture', {'type': 'play_pause'})
                            last_pinch_time = current_time
                            is_pinch_active = True
                            current_interaction_lock = 'pinch' # Ensure lock is set
                    elif not is_currently_pinching and is_pinch_active:
                        is_pinch_active = False
                        if current_interaction_lock == 'pinch':
                            current_interaction_lock = None
                            lock_cooldown_until = current_time + 0.65 # Cooldown on release

                    # Draw landmarks on frame for visual debug window
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            else:
                # Hand not detected: increment grace period frame counter
                tracking_loss_frames += 1
                if is_dragging and tracking_loss_frames <= 4:
                    # Do not release yet, wait to see if tracking returns!
                    continue

                # After 4 frames of continuous loss, finalize release
                tracking_loss_frames = 0
                if is_dragging:
                    # Exit Flick: hand exited view while dragging!
                    # Directional Boundary Projection (35% margin)
                    exit_margin = 0.35
                    if prev_x is not None and prev_y is not None:
                        is_moving_outward_x = (last_vx < -0.0005 and prev_x < exit_margin) or (last_vx > 0.0005 and prev_x > (1.0 - exit_margin))
                        is_moving_outward_y = (last_vy < -0.0005 and prev_y < exit_margin) or (last_vy > 0.0005 and prev_y > (1.0 - exit_margin))
                        was_near_border = is_moving_outward_x or is_moving_outward_y
                        
                        exit_speed = (last_vx**2 + last_vy**2)**0.5
                        speed_per_frame = exit_speed * 0.033 # convert to displacement per frame
                        
                        if was_near_border and speed_per_frame > 0.03:
                            direction_sign = 1 if last_vx >= 0 else -1
                            spin_speed = min(0.25, max(-0.25, direction_sign * exit_speed * 0.08))
                            print(f"[Camera Gesture] True Exit Flick detected! -> Spin speed: {spin_speed}")
                            sio.emit('hardware_gesture', {'type': 'swipe', 'value': spin_speed})
                        else:
                            # Central Release: apply velocity-based hybrid dampening to prevent unexpected high-speed spins!
                            direction_sign = 1 if last_vx >= 0 else -1
                            if speed_per_frame > 0.03:
                                spin_speed = min(0.06, max(-0.06, direction_sign * exit_speed * 0.02))
                            else:
                                spin_speed = min(0.02, max(-0.02, direction_sign * exit_speed * 0.01))
                            print(f"[Camera Gesture] Central dampened spin speed: {spin_speed}")
                            sio.emit('hardware_gesture', {'type': 'swipe', 'value': spin_speed})
                    is_dragging = False
                prev_x = None
                prev_y = None

            if os.environ.get('DISPLAY'):
                try:
                    cv2.imshow("Raspberry Pi Gesture Detector Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass
            else:
                # Headless SSH mode - no GUI display attached
                time.sleep(0.01)
    except Exception as e:
        print(f"[ERROR] Exception in camera loop: {e}")
    finally:
        if cap:
            cap.release()
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    use_camera = False
    if len(sys.argv) > 1 and sys.argv[1] == '--camera':
        use_camera = True

    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"Failed to connect to Flask server at {SERVER_URL}. Is app.py running?")
        print(e)
        sys.exit(1)
        
    try:
        if use_camera:
            run_physical_camera_detector()
        else:
            run_mock_keyboard_detector()
    except KeyboardInterrupt:
        pass
    finally:
        sio.disconnect()
