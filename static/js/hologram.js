document.addEventListener('DOMContentLoaded', () => {
    // 1. WebSocket Setup
    const socket = io();

    // DOM Elements with safe fallbacks for clean viewer pages
    const audio = document.getElementById('audio-player');
    const btnPlay = document.getElementById('btn-play') || document.createElement('button');
    const btnPrev = document.getElementById('btn-prev') || document.createElement('button');
    const btnNext = document.getElementById('btn-next') || document.createElement('button');
    const seekBar = document.getElementById('seek-bar') || document.createElement('input');
    const currentTimeEl = document.getElementById('current-time') || document.createElement('span');
    const durationEl = document.getElementById('duration') || document.createElement('span');
    const playlistEl = document.getElementById('playlist');

    const trackTitle = document.getElementById('track-title') || document.createElement('h2');
    const trackArtist = document.getElementById('track-artist') || document.createElement('p');
    const dashboardCover = document.getElementById('dashboard-cover') || document.createElement('img');
    const albumCoverContainer = document.querySelector('.album-cover-container') || document.createElement('div');

    const mainLayout = document.getElementById('main-layout') || document.createElement('div');
    const kioskBtn = document.getElementById('kiosk-mode-btn') || document.createElement('button');
    const frictionSlider = document.getElementById('friction-slider') || document.createElement('input');
    const sensitivitySlider = document.getElementById('sensitivity-slider') || document.createElement('input');
    const sensitivityValue = document.getElementById('sensitivity-value') || document.createElement('span');
    const guideToggle = document.getElementById('guide-toggle') || document.createElement('input');
    const webcamToggle = document.getElementById('webcam-toggle') || document.createElement('input');
    const btnGeminiCall = document.getElementById('btn-gemini-call');
    const btnTestListening = document.getElementById('btn-test-listening');
    const btnTestThinking = document.getElementById('btn-test-thinking');
    const btnTestSpeaking = document.getElementById('btn-test-speaking');
    const geminiChatSection = document.getElementById('gemini-chat-section');
    const geminiChatInput = document.getElementById('gemini-chat-input');
    const btnGeminiSend = document.getElementById('btn-gemini-send');
    const gestureStatus = document.getElementById('gesture-status') || document.createElement('p');
    const debugCanvas = document.getElementById('gesture-canvas') || { width: 200, height: 150 };
    const debugCtx = debugCanvas.getContext ? debugCanvas.getContext('2d') : {
        clearRect: () => { }, save: () => { }, restore: () => { },
        scale: () => { }, translate: () => { }, drawImage: () => { },
        beginPath: () => { }, arc: () => { }, fill: () => { }, stroke: () => { },
        closePath: () => { }, lineTo: () => { }, moveTo: () => { }
    };
    const videoElement = document.getElementById('webcam-video') || document.createElement('video');

    // State Variables
    let songs = [];
    let currentSongIndex = 0;
    let isPlaying = false;
    let audioCtx = null;
    let analyser = null;
    let dataArray = null;
    let sourceNode = null;

    // Interaction Parameters
    let friction = parseFloat(frictionSlider.value) || 0.98;
    let showGuides = guideToggle.checked;

    // Swipe & Rotation Physics State
    let isDragging = false;
    let previousX = 0;
    let previousY = 0;
    let dragVelocityX = 0;
    let dragVelocityY = 0;
    let lastTime = 0;

    // 3D rotation velocities
    let rotVelX = 0.001;
    let rotVelY = 0.005; // Auto-rotation speed when idle
    let rotVelZ = 0.001;
    let isGestureDragging = false; // Tracks if user is actively doing index drag rotation
    let isTouchActive = false;
    let handX = 0.5;
    let handY = 0.5;
    let currentCursorX = 0.5;
    let currentCursorY = 0.5;
    let currentInteractionLock = null; // Lock to ensure single active gesture mode
    let lockCooldownUntil = 0;          // Cooldown timestamp (ms) to prevent immediate re-triggering
    let interactionMode = 'rotate';     // Mode state machine: 'rotate' (cube) or 'music' (controls)
    let lastVx = 0;                     // Store last velocity for camera exit flick
    let lastVy = 0;
    let trackingLossFrames = 0;         // Frame buffer for high-speed tracking loss grace period
    let poseLossFrames = 0;             // Frame buffer for temporary pose misclassification grace period
    let dragSensitivity = 42.0;         // Interactive rotation sensitivity slider variable
    let lastInteractionTime = Date.now();

    // MediaPipe & Gesture Recognition State
    let handTracker = null;
    let cameraHelper = null;
    let mpFrameCount = 0; // Frame counter to throttle MediaPipe frames
    let seekStartAudioTime = 0; // Audio time when pinch-seek started
    let seekStartFingerX = 0;   // Finger X position when pinch-seek started
    let prevFingerX = null;
    let prevFingerY = null;
    let lastGestureTime = 0;
    let lastTrackChangeTime = 0;
    let lastPinchTime = 0;
    let isPinchingActive = false;
    let pinchStartTime = 0;
    let isSeekingActive = false;

    // Three.js Core Variables
    let renderer, scene, hologramCube, visualizerRing, lightCursor;
    let geminiGroup, geminiMesh, geminiParticles; // Gemini 3D Hologram variables
    let geminiVideo, geminiVideoTexture;          // Gemini video texture variables
    let isGeminiListening = false;
    let isGeminiThinking = false;
    let isGeminiSpeaking = false;
    let geminiRecognition = null;
    let currentUtterance = null; // Prevent SpeechSynthesisUtterance GC bug
    let wasPlayingBeforeGemini = false; // Save music playing state before voice session
    let geminiScalePulse = 1.0;
    let cameraNorth, cameraSouth, cameraEast, cameraWest;
    let textureLoader = new THREE.TextureLoader();
    let currentTexture = null;

    // Smooth transition target states
    let targetCubeOpacity = 1.0;
    let targetGeminiOpacity = 0.0;
    let targetCubeScale = 1.0;
    let targetGeminiScale = 0.0;

    let targetRotVelX = 0.001;
    let targetRotVelY = 0.005;
    let targetRotVelZ = 0.001;

    // --- State Machine Manager ---
    const HologramStateManager = {
        currentMode: 'STANDBY', // STANDBY, MUSIC, GEMINI
        subState: 'IDLE',       // STANDBY: IDLE | MUSIC: PLAYING, PAUSED, GESTURE_INTERACTING | GEMINI: LISTENING, THINKING, SPEAKING
        previousMode: 'STANDBY',
        previousSubState: 'IDLE',
        listeningTimeoutId: null,

        transitionTo(nextMode, nextSubState) {
            console.log(`[State Transition] ${this.currentMode}:${this.subState} -> ${nextMode}:${nextSubState}`);

            // 1. Exit Action
            this.exitStateAction(this.currentMode, this.subState);

            // 2. Backup previous state if entering GEMINI mode
            if (nextMode === 'GEMINI' && this.currentMode !== 'GEMINI') {
                this.previousMode = this.currentMode;
                this.previousSubState = this.subState;
            }

            // 3. Update values
            this.currentMode = nextMode;
            this.subState = nextSubState;

            // Legacy variable sync to avoid breaking other logic
            interactionMode = nextMode === 'STANDBY' ? 'rotate' : nextMode.toLowerCase();
            isGeminiListening = (nextSubState === 'LISTENING');
            isGeminiThinking = (nextSubState === 'THINKING');
            isGeminiSpeaking = (nextSubState === 'SPEAKING');

            // 4. Entry Action
            this.entryStateAction(nextMode, nextSubState);

            // 5. Trigger visual rendering hooks
            updateHologramRendering(nextMode, nextSubState);
        },

        entryStateAction(mode, state) {
            if (mode === 'GEMINI') {
                if (state === 'LISTENING') {
                    this.startListeningTimeout();
                    if (geminiRecognition) {
                        try {
                            geminiRecognition.abort();
                            geminiRecognition.start();
                        } catch (e) {
                            console.error("SpeechRecognition start error:", e);
                        }
                    }
                    if (geminiVideo) {
                        geminiVideo.currentTime = 0;
                        geminiVideo.play().catch(e => console.warn("Video play failed:", e));
                    }
                }
            }
        },

        exitStateAction(mode, state) {
            this.clearListeningTimeout();
            if (mode === 'GEMINI') {
                if (state === 'LISTENING') {
                    if (geminiRecognition) {
                        try { geminiRecognition.abort(); } catch (e) { }
                    }
                } else if (state === 'SPEAKING') {
                    stopTTS();
                }
            }
        },

        startListeningTimeout() {
            this.clearListeningTimeout();
            this.listeningTimeoutId = setTimeout(() => {
                console.log("[Timeout] No voice input detected in LISTENING. Restoring previous state...");
                this.restorePreviousState();
            }, 8000); // 8 seconds
        },

        clearListeningTimeout() {
            if (this.listeningTimeoutId) {
                clearTimeout(this.listeningTimeoutId);
                this.listeningTimeoutId = null;
            }
        },

        restorePreviousState() {
            if (wasPlayingBeforeGemini) {
                wasPlayingBeforeGemini = false; // reset
                this.transitionTo('MUSIC', 'PLAYING');
                playSong();
            } else {
                this.transitionTo(this.previousMode, this.previousSubState);
            }
        }
    };

    function updateHologramRendering(mode, state) {
        if (!hologramCube || !visualizerRing || !geminiGroup) return;

        // Force both groups to be visible during transition (so opacity fade calculations work)
        geminiGroup.visible = true;
        hologramCube.visible = true;
        if (visualizerRing) {
            visualizerRing.visible = true;
        }

        if (mode === 'STANDBY') {
            targetCubeOpacity = 1.0;
            targetCubeScale = 1.0;
            targetGeminiOpacity = 0.0;
            targetGeminiScale = 0.0;

            targetRotVelX = 0.001;
            targetRotVelY = 0.005;
            targetRotVelZ = 0.001;

            if (geminiVideo) geminiVideo.pause();
            if (geminiChatSection) geminiChatSection.style.display = 'none';
        }
        else if (mode === 'MUSIC') {
            targetCubeOpacity = 1.0;
            targetCubeScale = 1.0;
            targetGeminiOpacity = 0.0;
            targetGeminiScale = 0.0;

            if (geminiVideo) geminiVideo.pause();
            if (geminiChatSection) geminiChatSection.style.display = 'none';

            if (state === 'PLAYING') {
                targetRotVelX = 0.001;
                targetRotVelY = 0.015;
                targetRotVelZ = 0.001;
            } else if (state === 'PAUSED') {
                targetRotVelX = 0.0;
                targetRotVelY = 0.002;
                targetRotVelZ = 0.0;
            }
        }
        else if (mode === 'GEMINI') {
            targetCubeOpacity = 0.0;
            targetCubeScale = 0.0;
            targetGeminiOpacity = 1.0;
            targetGeminiScale = 1.0;

            if (geminiChatSection) {
                geminiChatSection.style.display = 'block';
                geminiChatInput.focus();
            }

            if (state === 'LISTENING') {
                targetRotVelX = 0.005;
                targetRotVelY = 0.015;
                targetRotVelZ = 0.0;
                if (geminiVideo) {
                    geminiVideo.play().catch(e => { });
                    geminiVideo.playbackRate = 1.0;
                }
            } else if (state === 'THINKING') {
                targetRotVelX = 0.05;
                targetRotVelY = 0.10;
                targetRotVelZ = 0.05;
                if (geminiVideo) {
                    geminiVideo.play().catch(e => { });
                    geminiVideo.playbackRate = 2.0;
                }
            } else if (state === 'SPEAKING') {
                targetRotVelX = 0.0;
                targetRotVelY = 0.008;
                targetRotVelZ = 0.0;
                if (geminiVideo) {
                    geminiVideo.play().catch(e => { });
                    geminiVideo.playbackRate = 1.0;
                }
            }
        }
    }

    // Initialize 3D Engine
    initThreeJS();

    // 2. Load Song Data
    fetch('/api/songs')
        .then(res => res.json())
        .then(data => {
            songs = data;
            buildPlaylist();
            loadSong(0);
        })
        .catch(err => console.error("Error loading songs:", err));

    function buildPlaylist() {
        if (!playlistEl) return;
        playlistEl.innerHTML = '';
        songs.forEach((song, idx) => {
            const item = document.createElement('div');
            item.className = `song-item ${idx === currentSongIndex ? 'active' : ''}`;
            item.innerHTML = `
                <img class="song-item-cover" src="${song.cover}" alt="cover">
                <div class="song-item-info">
                    <div class="song-item-title">${song.title}</div>
                    <div class="song-item-artist">${song.artist}</div>
                </div>
            `;
            item.addEventListener('click', () => {
                loadSong(idx);
                playSong();
            });
            playlistEl.appendChild(item);
        });
    }

    function loadSong(index) {
        currentSongIndex = index;
        const song = songs[index];
        audio.src = song.src;
        trackTitle.textContent = song.title;
        trackArtist.textContent = song.artist;
        dashboardCover.src = song.cover;

        // Update playlist UI highlight
        document.querySelectorAll('.song-item').forEach((item, idx) => {
            item.classList.toggle('active', idx === currentSongIndex);
        });

        // Load texture for 3D model
        textureLoader.load(song.cover, (texture) => {
            currentTexture = texture;
            if (hologramCube) {
                // Apply texture to front and back faces (materials index 4 and 5 in box mapping)
                const coverMaterial = new THREE.MeshBasicMaterial({
                    map: texture,
                    transparent: true,
                    opacity: targetCubeOpacity
                });

                hologramCube.material = [
                    coverMaterial, // right
                    coverMaterial, // left
                    coverMaterial, // top
                    coverMaterial, // bottom
                    coverMaterial, // front
                    coverMaterial  // back
                ];
            }
        });

        // Reset progress bar
        seekBar.value = 0;
        currentTimeEl.textContent = '0:00';
        setTimeout(syncPlayerState, 200);
    }

    // Audio Playback Actions
    function togglePlay() {
        if (isPlaying) {
            pauseSong();
        } else {
            playSong();
        }
    }

    function playSong() {
        setupAudioContext();
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        audio.play().then(() => {
            if (audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            isPlaying = true;
            btnPlay.textContent = '⏸️';
            albumCoverContainer.style.animationPlayState = 'running';
            HologramStateManager.transitionTo('MUSIC', 'PLAYING');
            syncPlayerState();
        }).catch(err => {
            console.error("Audio playback block:", err);
        });
    }

    function pauseSong() {
        audio.pause();
        isPlaying = false;
        btnPlay.textContent = '▶️';
        albumCoverContainer.style.animationPlayState = 'paused';
        HologramStateManager.transitionTo('MUSIC', 'PAUSED');
        syncPlayerState();
    }

    function prevSong(forcePlay = false) {
        let nextIndex = currentSongIndex - 1;
        if (nextIndex < 0) nextIndex = songs.length - 1;
        loadSong(nextIndex);
        if (isPlaying || forcePlay) playSong();
    }

    function nextSong(forcePlay = false) {
        let nextIndex = (currentSongIndex + 1) % songs.length;
        loadSong(nextIndex);
        if (isPlaying || forcePlay) playSong();
    }

    function setupAudioContext() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 64; // Small fft for lightweight reaction
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            sourceNode = audioCtx.createMediaElementSource(audio);
            sourceNode.connect(analyser);
            analyser.connect(audioCtx.destination);
        }
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    }

    window.addEventListener('click', () => {
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    });

    // Audio player event listeners
    btnPlay.addEventListener('click', togglePlay);
    btnPrev.addEventListener('click', () => prevSong());
    btnNext.addEventListener('click', () => nextSong());

    let lastSyncTime = 0;
    audio.addEventListener('timeupdate', () => {
        if (audio.duration) {
            const pct = (audio.currentTime / audio.duration) * 100;
            seekBar.value = pct;

            // Format time
            const curMins = Math.floor(audio.currentTime / 60);
            const curSecs = Math.floor(audio.currentTime % 60).toString().padStart(2, '0');
            const durMins = Math.floor(audio.duration / 60);
            const durSecs = Math.floor(audio.duration % 60).toString().padStart(2, '0');

            currentTimeEl.textContent = `${curMins}:${curSecs}`;
            durationEl.textContent = `${durMins}:${durSecs}`;

            // Sync with remote control (max 1 sync per second)
            const now = Date.now();
            if (now - lastSyncTime > 900) {
                lastSyncTime = now;
                if (typeof syncPlayerState === 'function') {
                    syncPlayerState();
                }
            }
        }
    });

    seekBar.addEventListener('input', () => {
        if (audio.duration) {
            audio.currentTime = (seekBar.value / 100) * audio.duration;
        }
    });

    // 3. Three.js Initialization (4-way Hologram viewports)
    function initThreeJS() {
        const canvas = document.getElementById('hologram-canvas');
        renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
        renderer.setSize(window.innerWidth, window.innerHeight);
        // Set pixel ratio to 1.0 for performance optimization on Raspberry Pi 5
        renderer.setPixelRatio(1.0);
        renderer.autoClear = false;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000); // Black room for hologram reflection

        // 3D Album Art Object (Cube representing album block)
        const geometry = new THREE.BoxGeometry(2.0, 2.0, 2.0);
        const coverMaterial = new THREE.MeshBasicMaterial({
            color: 0x222222,
            transparent: true,
            opacity: 1.0
        });
        const initialMaterial = [
            coverMaterial, coverMaterial, // right, left
            coverMaterial, coverMaterial, // top, bottom
            coverMaterial, coverMaterial  // front, back
        ];

        hologramCube = new THREE.Mesh(geometry, initialMaterial);
        scene.add(hologramCube);

        // Create visualizerRing (glowing particle ring)
        const particleCount = 80;
        const ringGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const radius = 2.0;

        for (let i = 0; i < particleCount; i++) {
            const angle = (i / particleCount) * Math.PI * 2;
            positions[i * 3] = Math.cos(angle) * radius;
            positions[i * 3 + 1] = 0; // flat on XZ plane
            positions[i * 3 + 2] = Math.sin(angle) * radius;
        }

        ringGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const ringMaterial = new THREE.PointsMaterial({
            color: 0x00acff,
            size: 0.1,
            transparent: true,
            opacity: 0.85,
            blending: THREE.AdditiveBlending
        });

        visualizerRing = new THREE.Points(ringGeometry, ringMaterial);
        // scene.add(visualizerRing); // Commented out to disable particle effects

        // --- Create Gemini 3D Hologram ---
        geminiGroup = new THREE.Group();
        geminiGroup.visible = false; // Hidden initially in music mode
        scene.add(geminiGroup);

        // Initialize HTML5 Video element for Gemini Video Texture
        geminiVideo = document.createElement('video');
        geminiVideo.src = '/static/assets/gemini_3d.mp4';
        geminiVideo.loop = true;
        geminiVideo.muted = true;
        geminiVideo.playsInline = true;
        geminiVideo.style.display = 'none';
        document.body.appendChild(geminiVideo);

        // Initialize Three.js VideoTexture
        geminiVideoTexture = new THREE.VideoTexture(geminiVideo);
        geminiVideoTexture.minFilter = THREE.LinearFilter;
        geminiVideoTexture.magFilter = THREE.LinearFilter;
        geminiVideoTexture.format = THREE.RGBAFormat;

        // Project video texture onto PlaneGeometry (representing the 3D rotating visual)
        const geminiPlaneGeo = new THREE.PlaneGeometry(2.2, 2.2);
        const geminiPlaneMat = new THREE.MeshBasicMaterial({
            map: geminiVideoTexture,
            transparent: true,
            opacity: 1.0,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending // Blends black background of MP4 out dynamically!
        });

        geminiMesh = new THREE.Mesh(geminiPlaneGeo, geminiPlaneMat);
        geminiGroup.add(geminiMesh);

        // Gemini Particle Halo
        const geminiPartCount = 120;
        const geminiPartGeo = new THREE.BufferGeometry();
        const geminiPartPositions = new Float32Array(geminiPartCount * 3);

        for (let i = 0; i < geminiPartCount; i++) {
            const angle = (i / geminiPartCount) * Math.PI * 2;
            const deviation = (Math.random() - 0.5) * 0.3;
            geminiPartPositions[i * 3] = Math.cos(angle) * (1.5 + deviation);
            geminiPartPositions[i * 3 + 1] = (Math.random() - 0.5) * 0.4;
            geminiPartPositions[i * 3 + 2] = Math.sin(angle) * (1.5 + deviation);
        }
        geminiPartGeo.setAttribute('position', new THREE.BufferAttribute(geminiPartPositions, 3));

        const geminiPartMat = new THREE.PointsMaterial({
            color: 0xff00d0, // bright pink
            size: 0.08,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending
        });

        geminiParticles = new THREE.Points(geminiPartGeo, geminiPartMat);
        // geminiGroup.add(geminiParticles); // Commented out to disable particle effects

        // Initialize Gemini group, mesh and particles to 0 scale and opacity for bulletproof transition startup
        geminiGroup.scale.set(0, 0, 0);
        geminiMesh.scale.set(0, 0, 0);
        geminiMesh.material.opacity = 0.0;
        geminiParticles.material.opacity = 0.0;

        // Create Holographic Light Cursor (Alternative 3)
        const cursorGeo = new THREE.RingGeometry(0.12, 0.22, 32);
        const cursorMat = new THREE.MeshBasicMaterial({
            color: 0xd000ff,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.0, // hidden until touch zone active
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });
        lightCursor = new THREE.Mesh(cursorGeo, cursorMat);
        hologramCube.add(lightCursor);

        // 4 Cameras facing the center from North, South, East, West
        const aspect = 1;
        const size = 3;
        const dist = 5.0; // Distance of cameras from center

        // cameraSouth: bottom viewport on screen (bottom of album points UP towards center)
        cameraSouth = new THREE.PerspectiveCamera(45, aspect, 0.1, 20);
        cameraSouth.position.set(0, 0, dist);
        cameraSouth.up.set(0, -1, 0);
        cameraSouth.lookAt(0, 0, 0);

        // cameraNorth: top viewport on screen (bottom of album points DOWN towards center)
        cameraNorth = new THREE.PerspectiveCamera(45, aspect, 0.1, 20);
        cameraNorth.position.set(0, 0, -dist);
        cameraNorth.up.set(0, 1, 0);
        cameraNorth.lookAt(0, 0, 0);

        // cameraWest: left viewport on screen (bottom of album points RIGHT towards center)
        cameraWest = new THREE.PerspectiveCamera(45, aspect, 0.1, 20);
        cameraWest.position.set(-dist, 0, 0);
        cameraWest.up.set(0, 0, 1);
        cameraWest.lookAt(0, 0, 0);

        // cameraEast: right viewport on screen (bottom of album points LEFT towards center)
        cameraEast = new THREE.PerspectiveCamera(45, aspect, 0.1, 20);
        cameraEast.position.set(dist, 0, 0);
        cameraEast.up.set(0, 0, 1);
        cameraEast.lookAt(0, 0, 0);

        // Handle resize
        window.addEventListener('resize', onWindowResize);
        onWindowResize();



        // Start Animation Loop
        renderer.setAnimationLoop(animate);
    }

    function onWindowResize() {
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Update aspect ratios of cameras if window changes
        [cameraNorth, cameraSouth, cameraEast, cameraWest].forEach(cam => {
            cam.aspect = 1; // Always keep viewports square
            cam.updateProjectionMatrix();
        });
    }

    // 4. Animation loop and 4-way viewport splitting
    function animate() {


        // A. Apply rotation physics (Inertia & Damping)
        if (!isDragging && !isGestureDragging) {
            if (HologramStateManager.currentMode === 'GEMINI') {
                // In Gemini mode, smoothly interpolate rotation speeds (acceleration/deceleration damping)
                rotVelX += (targetRotVelX - rotVelX) * 0.08;
                rotVelY += (targetRotVelY - rotVelY) * 0.08;
                rotVelZ += (targetRotVelZ - rotVelZ) * 0.08;
            } else {
                // Apply friction deceleration to all 3 axes
                rotVelX *= friction;
                rotVelY *= friction;
                rotVelZ *= friction;

                // Calculate current total speed & time since last interaction
                const speed = Math.sqrt(rotVelX * rotVelX + rotVelY * rotVelY + rotVelZ * rotVelZ);
                const timeSinceLastInteraction = Date.now() - lastInteractionTime;
                const realignmentThreshold = 0.012; // speed threshold below which we realign
                const idleDelay = 5000;              // 5 seconds delay before starting realignment

                if (speed < realignmentThreshold && timeSinceLastInteraction > idleDelay) {
                    // Smoothly decay X and Z speed to 0
                    rotVelX += (0 - rotVelX) * 0.05;
                    rotVelZ += (0 - rotVelZ) * 0.05;

                    // Realign Y rotation speed to targetRotVelY
                    rotVelY += (targetRotVelY - rotVelY) * 0.05;

                    // Smoothly spring back / re-align X and Z rotation angles to 0
                    hologramCube.rotation.x += (0 - hologramCube.rotation.x) * 0.005;
                    hologramCube.rotation.z += (0 - hologramCube.rotation.z) * 0.005;

                    // Keep Y rotation spinning
                    hologramCube.rotation.y += rotVelY;
                } else {
                    // Free 3D spin under inertia around world axes
                    hologramCube.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), rotVelY);
                    hologramCube.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), rotVelX);
                    hologramCube.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), rotVelZ);
                }
            }
        }

        // Slight vertical oscillation for premium 3D float look
        const floatOffset = Math.sin(Date.now() * 0.001) * 0.15;
        hologramCube.position.y = floatOffset;

        if (HologramStateManager.currentMode === 'GEMINI') {
            // Apply floating animation
            geminiGroup.position.y = floatOffset;

            // Spin elements dynamically using the smoothed rotVelY!
            geminiMesh.rotation.y += rotVelY;
            geminiParticles.rotation.y -= rotVelY * 0.75;

            if (HologramStateManager.subState === 'THINKING') {
                if (geminiVideo && geminiVideo.playbackRate !== 3.5) {
                    geminiVideo.playbackRate = 3.5;
                }
            } else {
                if (geminiVideo && geminiVideo.playbackRate !== 1.0) {
                    geminiVideo.playbackRate = 1.0;
                }
            }
        }

        // B. Handle Audio Frequency Analysis and Spatial touch feedback
        let targetScale = 1.0;
        if (isPlaying) {
            if (analyser) {
                analyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for (let i = 0; i < 8; i++) {
                    sum += dataArray[i];
                }
                const avg = sum / 8; // 0 to 255
                targetScale = 1.0 + (avg / 255.0) * 0.4;

                if (visualizerRing) {
                    // Combine audio pulse with spatial touch scaling
                    let spatialScale = 1.0;
                    let targetColor = new THREE.Color(0x00acff);

                    if (isTouchActive) {
                        spatialScale = 1.25;
                        targetColor.setHex(0xd000ff); // Magenta
                    } else {
                        spatialScale = 1.0;
                        targetColor.setHex(0x00acff); // Cyan
                    }
                    hologramCube.position.x = 0;
                    hologramCube.position.z = 0;

                    // Smoothly interpolate color
                    visualizerRing.material.color.lerp(targetColor, 0.1);

                    // Set particle size with audio pulse
                    visualizerRing.material.size = (0.08 + (avg / 255.0) * 0.08) * (isTouchActive ? 1.3 : 1.0);

                    // Smoothly scale the ring
                    const ringTargetScale = spatialScale * (1.0 + (avg / 255.0) * 0.15);
                    const currentRingScale = visualizerRing.scale.x + (ringTargetScale - visualizerRing.scale.x) * 0.1;
                    visualizerRing.scale.set(currentRingScale, currentRingScale, currentRingScale);
                }
            } else {
                targetScale = 1.0;
            }
        } else {
            targetScale = 0.75;
            if (visualizerRing) {
                // Return to default resting scale/color when paused
                const restColor = new THREE.Color(0x00acff);
                visualizerRing.material.color.lerp(restColor, 0.1);
                const restScale = visualizerRing.scale.x + (0.8 - visualizerRing.scale.x) * 0.1;
                visualizerRing.scale.set(restScale, restScale, restScale);
                visualizerRing.material.size = 0.08;
            }
        }

        // Smoothly interpolate current scale to target scale (animation transition)
        const scaleSpeed = isPlaying ? 0.08 : 0.04; // swell up faster, shrink down slightly slower
        const currentScale = hologramCube.scale.x + (targetScale * targetCubeScale - hologramCube.scale.x) * scaleSpeed;
        hologramCube.scale.set(currentScale, currentScale, currentScale);

        hologramCube.position.x = 0;
        hologramCube.position.z = 0;

        // Smoothly interpolate material opacities for fade transition
        if (hologramCube.material) {
            const materials = Array.isArray(hologramCube.material) ? hologramCube.material : [hologramCube.material];
            materials.forEach(mat => {
                if (mat) mat.opacity += (targetCubeOpacity - mat.opacity) * 0.04;
            });
        }
        if (visualizerRing && visualizerRing.material) {
            visualizerRing.material.opacity += (targetCubeOpacity - visualizerRing.material.opacity) * 0.04;
        }
        if (geminiMesh && geminiMesh.material) {
            geminiMesh.material.opacity += (targetGeminiOpacity - geminiMesh.material.opacity) * 0.04;
        }
        if (geminiParticles && geminiParticles.material) {
            geminiParticles.material.opacity += (targetGeminiOpacity - geminiParticles.material.opacity) * 0.04;
        }

        // Smooth scale interpolation for parent Gemini group (runs every frame to support exit/entry transitions)
        if (geminiGroup) {
            const curGroupScale = geminiGroup.scale.x + (targetGeminiScale - geminiGroup.scale.x) * 0.05;
            geminiGroup.scale.set(curGroupScale, curGroupScale, curGroupScale);
        }

        // Smooth scale interpolation for child Gemini mesh (handles TTS voice boundary pulsing bounce)
        if (geminiMesh) {
            let targetGeminiScaleVal = 1.0;
            if (HologramStateManager.currentMode === 'GEMINI' && HologramStateManager.subState === 'SPEAKING') {
                // Decay the pulse trigger back to normal size smoothly
                geminiScalePulse += (1.0 - geminiScalePulse) * 0.1;
                targetGeminiScaleVal = geminiScalePulse;
            }
            const curMeshScale = geminiMesh.scale.x + (targetGeminiScaleVal - geminiMesh.scale.x) * 0.15;
            geminiMesh.scale.set(curMeshScale, curMeshScale, curMeshScale);
        }

        // Optimize visibility to disable GPU draw calls for transparent meshes
        const cubeMatOpacity = Array.isArray(hologramCube.material) ? hologramCube.material[0].opacity : hologramCube.material.opacity;
        hologramCube.visible = (targetCubeOpacity > 0.01 || cubeMatOpacity > 0.01);
        if (visualizerRing) {
            visualizerRing.visible = hologramCube.visible;
        }
        if (geminiGroup && geminiMesh && geminiMesh.material) {
            geminiGroup.visible = (targetGeminiOpacity > 0.01 || geminiMesh.material.opacity > 0.01);
        }

        // Update Holographic Light Cursor projection (Alternative 3)
        if (isTouchActive && lightCursor && interactionMode === 'rotate') {
            // Fade in cursor
            lightCursor.material.opacity += (0.9 - lightCursor.material.opacity) * 0.15;

            // Lerp cursor target to smooth out MediaPipe tracking jitter
            currentCursorX += (handX - currentCursorX) * 0.25;
            currentCursorY += (handY - currentCursorY) * 0.25;

            // Map [0, 1] screen coordinates to world coordinates
            const worldCursorX = (currentCursorX - 0.5) * 4.0;
            const worldCursorY = (0.5 - currentCursorY) * 4.0;
            const worldCursorZ = 1.5;

            const worldPos = new THREE.Vector3(worldCursorX, worldCursorY, worldCursorZ);

            // Convert to cube local space
            const localPos = hologramCube.worldToLocal(worldPos);

            // Project onto virtual bounding sphere of radius 1.45 to prevent mesh clipping
            localPos.normalize().multiplyScalar(1.45);

            lightCursor.position.copy(localPos);
            lightCursor.lookAt(0, 0, 0); // Keep tangent facing outward

            // Sync cursor color and pulse scale depending on touch vs grab state
            lightCursor.material.color.setHex(0xd000ff); // Magenta
            const pulse = 1.0 + Math.sin(Date.now() * 0.015) * 0.1;
            lightCursor.scale.set(pulse, pulse, pulse);
        } else if (lightCursor) {
            // Fade out cursor when inactive
            lightCursor.material.opacity += (0.0 - lightCursor.material.opacity) * 0.15;
        }

        // Rotate visualizer ring slowly in opposite direction
        if (visualizerRing) {
            visualizerRing.rotation.y = -hologramCube.rotation.y * 0.3;
            visualizerRing.position.y = floatOffset;
        }

        // C. Divide WebGL context based on View Mode (QUAD 4-Face vs SINGLE 1-Face East)
        const w = window.innerWidth;
        const h = window.innerHeight;

        const cx = w / 2;
        const cy = h / 2;

        renderer.setScissorTest(false);
        renderer.clear();
        renderer.setScissorTest(true);

        if (window.hologramViewMode === 'SINGLE_EAST') {
            // Single 1-Face Mode (East View shifted to right side with custom size)
            const scale = window.singleModeScale || 0.48; // default 48% size
            const offsetXRatio = (window.singleModeOffsetX !== undefined) ? window.singleModeOffsetX : 0.30; // shifted to East side
            const singleSize = Math.min(w, h) * scale;
            const vx = cx + (Math.min(w, h) * offsetXRatio) - (singleSize / 2);
            const vy = cy - (singleSize / 2);

            renderer.setViewport(vx, vy, singleSize, singleSize);
            renderer.setScissor(vx, vy, singleSize, singleSize);
            renderer.render(scene, cameraEast);
        } else {
            // Quad 4-Face Pyramid Mode (Default)
            const viewSize = Math.min(w, h) * 0.28;

            // 1. SOUTH (Bottom view)
            renderer.setViewport(cx - viewSize / 2, cy - viewSize * 1.5, viewSize, viewSize);
            renderer.setScissor(cx - viewSize / 2, cy - viewSize * 1.5, viewSize, viewSize);
            renderer.render(scene, cameraSouth);

            // 2. NORTH (Top view)
            renderer.setViewport(cx - viewSize / 2, cy + viewSize * 0.5, viewSize, viewSize);
            renderer.setScissor(cx - viewSize / 2, cy + viewSize * 0.5, viewSize, viewSize);
            renderer.render(scene, cameraNorth);

            // 3. WEST (Left view)
            renderer.setViewport(cx - viewSize * 1.5, cy - viewSize / 2, viewSize, viewSize);
            renderer.setScissor(cx - viewSize * 1.5, cy - viewSize / 2, viewSize, viewSize);
            renderer.render(scene, cameraWest);

            // 4. EAST (Right view)
            renderer.setViewport(cx + viewSize * 0.5, cy - viewSize / 2, viewSize, viewSize);
            renderer.setScissor(cx + viewSize * 0.5, cy - viewSize / 2, viewSize, viewSize);
            renderer.render(scene, cameraEast);

            // Draw crosshair alignment guides if enabled
            if (showGuides) {
                drawCalibrationGuides(cx, cy, viewSize);
            }
        }
    }

    // Helper: Draw crosshair alignment guides on the canvas overlay
    function drawCalibrationGuides(cx, cy, size) {
        // We can draw directly on WebGL or use standard 2D overlay.
        // For efficiency, let's keep guides drawn using viewport scissor or a lightweight overlay.
        // As a simpler method, we will inject a clean inline Canvas 2D overlay if guides are checked!
        // To keep it simple, we draw it using Three.js lines or a separate canvas, but here we can just skip drawing or use CSS lines.
        // Let's implement lightweight CSS guide lines in index.html, which is cleaner.
        // We will add/remove a helper class to the container.
    }

    // 5. Drag & Swipe Interaction Logic (Mouse & Touch)
    const canvasEl = document.getElementById('hologram-canvas');

    canvasEl.addEventListener('mousedown', startDrag);
    canvasEl.addEventListener('mousemove', drag);
    window.addEventListener('mouseup', endDrag);

    canvasEl.addEventListener('touchstart', (e) => startDrag(e.touches[0]));
    canvasEl.addEventListener('touchmove', (e) => drag(e.touches[0]));
    window.addEventListener('touchend', endDrag);

    function startDrag(e) {
        // Check if user is clicking in the TOP SEEK ZONE (top 15% of screen height)
        const clickY = e.clientY;
        const screenH = window.innerHeight;

        if (clickY < screenH * 0.15) {
            // Seek bar interaction simulated
            const clickX = e.clientX;
            const screenW = window.innerWidth;
            if (audio.duration) {
                audio.currentTime = (clickX / screenW) * audio.duration;
            }
            return;
        }

        isDragging = true;
        previousX = e.clientX;
        previousY = e.clientY;
        lastTime = performance.now();
        dragVelocityX = 0;
        dragVelocityY = 0;
    }

    function drag(e) {
        if (!isDragging) return;

        const currentX = e.clientX;
        const currentY = e.clientY;
        const now = performance.now();
        const dt = now - lastTime;
        const dx = currentX - previousX;
        const dy = currentY - previousY;

        if (dt > 0) {
            // Calculate instantaneous drag velocity
            dragVelocityX = dx / dt;
            dragVelocityY = dy / dt;
            // Rotate cube instantly during drag on multiple axes (aligned to match physical direction)
            hologramCube.rotation.y += dx * 0.007;
            hologramCube.rotation.x += dy * 0.007;
            hologramCube.rotation.z += (dx + dy) * 0.003;
            lastInteractionTime = Date.now();
        }

        previousX = currentX;
        previousY = currentY;
        lastTime = now;
    }

    function endDrag() {
        if (!isDragging) return;
        isDragging = false;

        // Map drag velocity to target rotation velocity (aligned to match physical direction)
        rotVelY = dragVelocityX * 0.8;
        rotVelX = dragVelocityY * 0.8;
        rotVelZ = (dragVelocityX + dragVelocityY) * 0.3;

        const maxVel = 0.2;
        rotVelY = Math.max(-maxVel, Math.min(maxVel, rotVelY));
        rotVelX = Math.max(-maxVel, Math.min(maxVel, rotVelX));
        rotVelZ = Math.max(-maxVel, Math.min(maxVel, rotVelZ));

        lastInteractionTime = Date.now();
    }

    // Double tap/click to play/pause in kiosk mode
    let lastTap = 0;
    canvasEl.addEventListener('click', (e) => {
        const now = Date.now();
        if (now - lastTap < 300) {
            togglePlay();
        }
        lastTap = now;
    });

    // 6. Settings controls
    frictionSlider.addEventListener('input', () => {
        friction = parseFloat(frictionSlider.value);
    });

    if (sensitivitySlider && sensitivityValue) {
        sensitivitySlider.addEventListener('input', (e) => {
            dragSensitivity = parseFloat(e.target.value);
            sensitivityValue.textContent = Math.round(dragSensitivity);
        });
    }

    if (btnGeminiCall) {
        btnGeminiCall.addEventListener('click', () => {
            toggleGeminiMode();
        });
    }

    if (btnTestListening) {
        btnTestListening.addEventListener('click', () => {
            wasPlayingBeforeGemini = isPlaying;
            if (isPlaying) pauseSong();
            HologramStateManager.transitionTo('GEMINI', 'LISTENING');
        });
    }
    if (btnTestThinking) {
        btnTestThinking.addEventListener('click', () => {
            wasPlayingBeforeGemini = isPlaying;
            if (isPlaying) pauseSong();
            HologramStateManager.transitionTo('GEMINI', 'THINKING');
        });
    }
    if (btnTestSpeaking) {
        btnTestSpeaking.addEventListener('click', () => {
            wasPlayingBeforeGemini = isPlaying;
            if (isPlaying) pauseSong();
            HologramStateManager.transitionTo('GEMINI', 'SPEAKING');
        });
    }

    function submitGeminiChat() {
        const text = geminiChatInput.value.trim();
        if (!text) return;

        geminiChatInput.value = '';

        // Abort speech recognition to prevent overlap
        if (geminiRecognition) {
            try { geminiRecognition.abort(); } catch (e) { }
        }

        // Transition state and call API
        HologramStateManager.transitionTo('GEMINI', 'THINKING');
        sendToGemini(text);
    }

    if (btnGeminiSend) {
        btnGeminiSend.addEventListener('click', submitGeminiChat);
    }
    if (geminiChatInput) {
        geminiChatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                submitGeminiChat();
            }
        });
    }

    // Create / Remove CSS Grid Lines based on Guide Toggle
    let guideLinesContainer = null;
    function toggleGuideLines() {
        showGuides = guideToggle.checked;
        if (showGuides) {
            if (!guideLinesContainer) {
                guideLinesContainer = document.createElement('div');
                guideLinesContainer.className = 'guide-lines';
                guideLinesContainer.innerHTML = `
                    <div style="position: absolute; top: 50%; left: 0; width: 100%; height: 1px; border-top: 1px dashed rgba(255,255,255,0.2); pointer-events: none;"></div>
                    <div style="position: absolute; top: 0; left: 50%; width: 1px; height: 100%; border-left: 1px dashed rgba(255,255,255,0.2); pointer-events: none;"></div>
                    <div style="position: absolute; top: 50%; left: 50%; width: 100px; height: 100px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none;"></div>
                `;
                document.querySelector('.hologram-viewport-area').appendChild(guideLinesContainer);
            }
            guideLinesContainer.style.display = 'block';
        } else {
            if (guideLinesContainer) {
                guideLinesContainer.style.display = 'none';
            }
        }
    }
    guideToggle.addEventListener('change', toggleGuideLines);
    toggleGuideLines(); // Init

    // Fullscreen Kiosk toggles
    kioskBtn.addEventListener('click', () => {
        mainLayout.classList.add('kiosk-active');
        onWindowResize();
    });

    // Exit Kiosk Mode on double tap on hologram area
    document.querySelector('.hologram-viewport-area').addEventListener('dblclick', () => {
        mainLayout.classList.remove('kiosk-active');
        onWindowResize();
    });

    // 7. Local Webcam Hand Gestures using MediaPipe Hands
    webcamToggle.addEventListener('change', () => {
        if (webcamToggle.checked) {
            initMediaPipe();
        } else {
            stopMediaPipe();
        }
    });

    function initMediaPipe() {
        gestureStatus.textContent = 'Initializing Camera...';

        // Initialize MediaPipe Hands object
        handTracker = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        handTracker.setOptions({
            maxNumHands: 1,
            modelComplexity: 0, // Lite model for faster CPU/GPU inference on Raspberry Pi
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        handTracker.onResults(onHandResults);

        // Access webcam
        mpFrameCount = 0;
        navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
            .then(stream => {
                videoElement.srcObject = stream;
                cameraHelper = new Camera(videoElement, {
                    onFrame: async () => {
                        if (webcamToggle.checked) {
                            await handTracker.send({ image: videoElement });
                        }
                    },
                    width: 320,
                    height: 240
                });
                cameraHelper.start();
                gestureStatus.textContent = '웹캠 감지 중 (손가락을 보여주세요)';
            })
            .catch(err => {
                console.error("Camera access failed:", err);
                gestureStatus.textContent = '카메라 연결 실패';
                webcamToggle.checked = false;
            });
    }

    function stopMediaPipe() {
        gestureStatus.textContent = '제스처 비활성화';
        if (cameraHelper) {
            cameraHelper.stop();
            cameraHelper = null;
        }
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }
        debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
    }

    function onHandResults(results) {
        debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);

        // Draw mirror-image frame
        debugCtx.save();
        debugCtx.scale(-1, 1);
        debugCtx.translate(-debugCanvas.width, 0);

        if (results.image) {
            debugCtx.drawImage(results.image, 0, 0, debugCanvas.width, debugCanvas.height);
        }
        debugCtx.restore();

        if (HologramStateManager.currentMode === 'GEMINI') {
            return;
        }

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            trackingLossFrames = 0; // reset grace frame buffer
            const landmarks = results.multiHandLandmarks[0];
            drawHandLandmarks(landmarks);

            // Index finger tip (Landmark 8), wrist (Landmark 0) & index knuckle (Landmark 5)
            const indexTip = landmarks[8];
            const wrist = landmarks[0];
            const indexKnuckle = landmarks[5];

            // 1. Proximity Check (2D distance between wrist and index knuckle in normalized image space)
            const handScale = Math.sqrt(
                Math.pow(wrist.x - indexKnuckle.x, 2) +
                Math.pow(wrist.y - indexKnuckle.y, 2)
            );

            if (handScale < 0.11) {
                gestureStatus.textContent = '손을 카메라에 더 가까이 대주세요 (Too Far)';
                prevFingerX = null;
                prevFingerY = null;
                isPinchingActive = false;
                isSeekingActive = false;
                isPinchArmed = false;
                isSkipArmed = false;
                return;
            }

            // 2. Finger extension check (comparing Tip distance vs PIP joint distance from Wrist)
            // Index finger (tip 8, PIP 6)
            const d_wrist_index_pip = Math.sqrt(
                Math.pow(wrist.x - landmarks[6].x, 2) +
                Math.pow(wrist.y - landmarks[6].y, 2) +
                Math.pow(wrist.z - landmarks[6].z, 2)
            );
            const d_wrist_index_tip = Math.sqrt(
                Math.pow(wrist.x - indexTip.x, 2) +
                Math.pow(wrist.y - indexTip.y, 2) +
                Math.pow(wrist.z - indexTip.z, 2)
            );
            const isIndexExtended = d_wrist_index_tip > d_wrist_index_pip;

            // Middle finger (tip 12, PIP 10)
            const d_wrist_middle_pip = Math.sqrt(
                Math.pow(wrist.x - landmarks[10].x, 2) +
                Math.pow(wrist.y - landmarks[10].y, 2) +
                Math.pow(wrist.z - landmarks[10].z, 2)
            );
            const d_wrist_middle_tip = Math.sqrt(
                Math.pow(wrist.x - landmarks[12].x, 2) +
                Math.pow(wrist.y - landmarks[12].y, 2) +
                Math.pow(wrist.z - landmarks[12].z, 2)
            );
            const isMiddleExtended = d_wrist_middle_tip > d_wrist_middle_pip;

            // Ring finger (tip 16, PIP 14)
            const d_wrist_ring_pip = Math.sqrt(
                Math.pow(wrist.x - landmarks[14].x, 2) +
                Math.pow(wrist.y - landmarks[14].y, 2) +
                Math.pow(wrist.z - landmarks[14].z, 2)
            );
            const d_wrist_ring_tip = Math.sqrt(
                Math.pow(wrist.x - landmarks[16].x, 2) +
                Math.pow(wrist.y - landmarks[16].y, 2) +
                Math.pow(wrist.z - landmarks[16].z, 2)
            );
            const isRingExtended = d_wrist_ring_tip > d_wrist_ring_pip;

            // Pinky finger (tip 20, PIP 18)
            const d_wrist_pinky_pip = Math.sqrt(
                Math.pow(wrist.x - landmarks[18].x, 2) +
                Math.pow(wrist.y - landmarks[18].y, 2) +
                Math.pow(wrist.z - landmarks[18].z, 2)
            );
            const d_wrist_pinky_tip = Math.sqrt(
                Math.pow(wrist.x - landmarks[20].x, 2) +
                Math.pow(wrist.y - landmarks[20].y, 2) +
                Math.pow(wrist.z - landmarks[20].z, 2)
            );
            const isPinkyExtended = d_wrist_pinky_tip > d_wrist_pinky_pip;

            // V-Sign represents index & middle finger extended, others folded
            const isVSign = isIndexExtended && isMiddleExtended && !isRingExtended && !isPinkyExtended;
            // Pointing finger represents ONLY index finger extended, others folded
            const isPointingFinger = isIndexExtended && !isMiddleExtended && !isRingExtended && !isPinkyExtended;
            // Fist/Grab represents all 4 fingers folded
            const isFist = !isIndexExtended && !isMiddleExtended && !isRingExtended && !isPinkyExtended;
            // Flat palm represents all 4 fingers extended
            const isFlatPalm = isIndexExtended && isMiddleExtended && isRingExtended && isPinkyExtended;



            // Choose tracking cursor based on pose (Always indexTip now since grab/fist is removed)
            const cursorLandmark = indexTip;
            const x = 1.0 - cursorLandmark.x;
            const y = cursorLandmark.y;

            // Save absolute coordinates to global state for ThreeJS lightCursor projection
            handX = x;
            handY = y;

            // Define Edge Zone check (8% border margin)
            const edgeMargin = 0.08;
            const isAtEdge = (x < edgeMargin || x > (1.0 - edgeMargin) || y < edgeMargin || y > (1.0 - edgeMargin));

            // Mode Transition State Machine (ONLY if hand is NOT at the edge or we already have a lock)
            if (!isAtEdge || currentInteractionLock !== null) {
                if (isPointingFinger) {
                    if (interactionMode !== 'rotate') {
                        interactionMode = 'rotate';
                        prevFingerX = null;
                        prevFingerY = null;
                        isPinchingActive = false;
                        isSeekingActive = false;
                    }
                } else if (isFlatPalm) {
                    if (interactionMode !== 'music') {
                        interactionMode = 'music';
                        prevFingerX = null;
                        prevFingerY = null;
                        isGestureDragging = false;
                        isTouchActive = false;
                    }
                }
            }

            // Touch Active threshold: Z-depth proximity
            isTouchActive = handScale >= 0.14;

            // 3. Pinch detection (Thumb tip = 4, Index tip = 8)
            const thumbTip = landmarks[4];
            const pinchDist = Math.sqrt(
                Math.pow(thumbTip.x - indexTip.x, 2) +
                Math.pow(thumbTip.y - indexTip.y, 2)
            );
            const isCurrentlyPinching = pinchDist < 0.08 && !isFist;

            // --- Mutex State Machine Lock with Cooldown ---
            let lockReleased = false;

            if (currentInteractionLock === 'pinch') {
                if (!isCurrentlyPinching && !isPinchingActive) {
                    currentInteractionLock = null;
                    lockReleased = true;
                }
            } else if (currentInteractionLock === 'swipe') {
                if (!isVSign) {
                    currentInteractionLock = null;
                    lockReleased = true;
                }
            } else if (currentInteractionLock === 'rotate') {
                const isPoseStillValid = isTouchActive && isPointingFinger && interactionMode === 'rotate';
                if (isPoseStillValid) {
                    poseLossFrames = 0; // Reset grace counter
                } else {
                    if (interactionMode !== 'rotate') {
                        // User explicitly changed mode to music via palm: release instantly!
                        currentInteractionLock = null;
                        lockReleased = true;
                        poseLossFrames = 0;
                    } else {
                        // Temp pose loss during fast motion: increment buffer
                        poseLossFrames++;
                        if (poseLossFrames > 6) {
                            currentInteractionLock = null;
                            lockReleased = true;
                            poseLossFrames = 0;
                        }
                    }
                }
            }

            // If lock was released, trigger a 650ms cooldown before another lock can be acquired
            if (lockReleased) {
                lockCooldownUntil = Date.now() + 650;
            }

            // Assign new lock ONLY if not in cooldown AND hand is not at the edge (ignores edge entries)
            if (currentInteractionLock === null && Date.now() >= lockCooldownUntil && !isAtEdge) {
                if (isCurrentlyPinching && interactionMode === 'music') {
                    currentInteractionLock = 'pinch';
                } else if (isVSign && interactionMode === 'music') {
                    currentInteractionLock = 'swipe';
                } else if (isTouchActive && isPointingFinger && interactionMode === 'rotate') {
                    currentInteractionLock = 'rotate';
                    // Reset baselines for rotation tracking
                    prevFingerX = x;
                    prevFingerY = y;
                }
            }

            // Show mode header/status
            let modeText = interactionMode === 'rotate'
                ? '🔄 [큐브 제어 모드] - 검지로 터치/회전'
                : '🎵 [음악 제어 모드] - 핀치(재생) / V자(곡넘김)';

            // 4. Branch execution based on Mutex lock
            if (currentInteractionLock === 'pinch' && interactionMode === 'music') {
                // Handle pinch release first
                if (!isCurrentlyPinching && isPinchingActive) {
                    isPinchingActive = false;
                    if (!isSeekingActive) {
                        const now = Date.now();
                        if (now - lastPinchTime > 1500) {
                            lastPinchTime = now;
                            gestureStatus.textContent = '제스처: 재생/일시정지 토글 ⏸️';
                            togglePlay();
                        }
                    }
                    isSeekingActive = false;
                    prevFingerX = null;
                    prevFingerY = null;
                    currentInteractionLock = null; // release lock on completion
                    lockCooldownUntil = Date.now() + 650; // set cooldown on release
                    return;
                }

                // Handle active pinch
                if (isCurrentlyPinching) {
                    if (!isPinchingActive) {
                        isPinchingActive = true;
                        pinchStartTime = Date.now();
                        isSeekingActive = false;
                    }

                    const pinchDuration = Date.now() - pinchStartTime;
                    const SEEK_START_DELAY = 400; // Hold pinch for 400ms to activate seek

                    if (pinchDuration >= SEEK_START_DELAY) {
                        if (!isSeekingActive) {
                            isSeekingActive = true;
                            seekStartAudioTime = audio.currentTime;
                            seekStartFingerX = x;
                        }
                        if (audio.duration) {
                            const sensitivity = 0.6;
                            const deltaX = x - seekStartFingerX;
                            let targetTime = seekStartAudioTime + (deltaX * sensitivity * audio.duration);
                            targetTime = Math.max(0, Math.min(audio.duration, targetTime));
                            audio.currentTime = targetTime;
                            gestureStatus.textContent = `재생 위치 조정 중 (${deltaX >= 0 ? '+' : ''}${Math.round(deltaX * 100)}% 이동, 현재: ${Math.round(targetTime)}초)`;
                        }
                    } else {
                        gestureStatus.textContent = '핀치를 유지하여 재생바 조절 모드로 전환...';
                    }

                    prevFingerX = null;
                    prevFingerY = null;
                    return;
                }
            }

            // If hand is at edge and not interacting, block interaction initiation
            if (isAtEdge && currentInteractionLock === null) {
                gestureStatus.textContent = `${modeText} - [가장자리 경계 감지] 손을 중앙으로 이동해 조작하세요 ⚠️`;
                prevFingerX = null;
                prevFingerY = null;
                isPinchingActive = false;
                isSeekingActive = false;
                isGestureDragging = false;
                isTouchActive = false;
                return;
            }

            // Allowed configurations
            const allowSwipe = (currentInteractionLock === 'swipe' || currentInteractionLock === null) && Date.now() >= lockCooldownUntil && interactionMode === 'music';
            const allowRotate = (currentInteractionLock === 'rotate' || currentInteractionLock === null) && Date.now() >= lockCooldownUntil && interactionMode === 'rotate';

            // Handle hand validation (only active when showing recognized gestures in their allowed modes)
            // If we are currently in the poseLoss grace period for rotation, we bypass validation failure!
            const isPoseLossGraceActive = (currentInteractionLock === 'rotate' && poseLossFrames > 0);
            const hasValidPose = (isVSign && allowSwipe) || (isPointingFinger && allowRotate) || (isFlatPalm && interactionMode === 'music') || isPoseLossGraceActive;
            if (!hasValidPose) {
                gestureStatus.textContent = Date.now() < lockCooldownUntil
                    ? `${modeText} - 전환 대기 중... (제스처 쿨다운 적용)`
                    : `${modeText} - 검지(회전 진입) 또는 손바닥(음악 제어 진입)을 보여주세요`;
                prevFingerX = null;
                prevFingerY = null;
                isPinchingActive = false;
                isSeekingActive = false;
                isGestureDragging = false;
                isTouchActive = false;
                return;
            }

            // Track dragging state for 3D rotation (only when rotating lock is active)
            isGestureDragging = (isTouchActive || isPoseLossGraceActive) && (isPointingFinger || isPoseLossGraceActive) && (currentInteractionLock === 'rotate') && (interactionMode === 'rotate');

            // Set initial drag baseline immediately to prevent coordinate jumping!
            if (isGestureDragging) {
                if (prevFingerX === null || prevFingerY === null) {
                    prevFingerX = x;
                    prevFingerY = y;
                }
            }

            // Calculate velocity (freeze updates during poseLoss grace frames to prevent garbage movement)
            let vx = 0;
            let vy = 0;
            if (prevFingerX !== null && prevFingerY !== null && !isPoseLossGraceActive) {
                const dx = x - prevFingerX;
                const dy = y - prevFingerY;
                const now = Date.now();
                const dt = now - lastGestureTime;
                if (dt > 10) {
                    vx = dx / dt;
                    vy = dy / dt;
                }
            }

            // Save velocities for exit flick
            if (isGestureDragging && !isPoseLossGraceActive && (vx !== 0 || vy !== 0)) {
                lastVx = vx;
                lastVy = vy;
            }
            processGestures(
                x, y, vx, vy,
                isVSign && allowSwipe,
                isPointingFinger && allowRotate && !isPoseLossGraceActive,
                false // isFist is always false now
            );

            // Freeze coordinate baseline updates during poseLoss grace period
            if (!isPoseLossGraceActive) {
                prevFingerX = x;
                prevFingerY = y;
            }
            lastGestureTime = Date.now();
        } else {
            // Hand not detected: increment grace period frame counter
            trackingLossFrames++;
            if (isGestureDragging && trackingLossFrames <= 4) {
                // Keep the drag state alive for up to 4 frames (do not release yet!)
                gestureStatus.textContent = '트래킹 일시 보정 중... (가려짐 방지)';
                return;
            }

            // After 4 frames of continuous loss, finalize the release/flick spin!
            trackingLossFrames = 0; // reset counter
            if (isGestureDragging) {
                // Directional Boundary Projection (35% margin)
                // Checks if hand is on a screen side and moving outward towards that boundary
                const exitMargin = 0.35;
                const isMovingOutwardX = (lastVx < -0.0005 && handX < exitMargin) || (lastVx > 0.0005 && handX > (1.0 - exitMargin));
                const isMovingOutwardY = (lastVy < -0.0005 && handY < exitMargin) || (lastVy > 0.0005 && handY > (1.0 - exitMargin));
                const wasNearBorder = isMovingOutwardX || isMovingOutwardY;

                const exitSpeed = Math.sqrt(lastVx * lastVx + lastVy * lastVy);
                const speedPerFrame = exitSpeed * 16.67; // convert to displacement per frame

                if (wasNearBorder && speedPerFrame > 0.03) {
                    const maxInertia = 0.25; // higher cap for exit flick
                    rotVelY = Math.max(-maxInertia, Math.min(maxInertia, lastVx * 25.0 * 18.0));
                    rotVelX = Math.max(-maxInertia, Math.min(maxInertia, lastVy * 25.0 * 18.0));
                    rotVelZ = Math.max(-maxInertia * 0.3, Math.min(maxInertia * 0.3, (lastVx + lastVy) * 25.0 * 3.0));
                    lastInteractionTime = Date.now();
                    gestureStatus.textContent = '손이 화면 밖으로 나감: 고속 회전 관성 적용 ☄️';
                } else {
                    // Central Release: apply velocity-based hybrid dampening to prevent unexpected high-speed spins!
                    if (speedPerFrame > 0.03) {
                        // Moderate spin if they flicked in the center
                        rotVelY = Math.max(-0.06, Math.min(0.06, lastVx * 25.0 * 4.0));
                        rotVelX = Math.max(-0.06, Math.min(0.06, lastVy * 25.0 * 4.0));
                        rotVelZ = Math.max(-0.018, Math.min(0.018, (lastVx + lastVy) * 25.0 * 0.6));
                        gestureStatus.textContent = '드래그 해제됨 (중앙 속도 감쇄 회전)';
                    } else {
                        // Very slow release if they slow down or stopped
                        rotVelY = Math.max(-0.02, Math.min(0.02, lastVx * 25.0 * 1.5));
                        rotVelX = Math.max(-0.02, Math.min(0.02, lastVy * 25.0 * 1.5));
                        rotVelZ = Math.max(-0.006, Math.min(0.006, (lastVx + lastVy) * 25.0 * 0.25));
                        gestureStatus.textContent = '드래그 해제됨 (중앙 가려짐 보정)';
                    }
                    lastInteractionTime = Date.now();
                }
            }
            prevFingerX = null;
            prevFingerY = null;
            isPinchingActive = false;
            isSeekingActive = false;
            isGestureDragging = false;
            isTouchActive = false;
        }
    }

    function processGestures(x, y, vx, vy, isVSign, isPointingFinger, isFist) {
        // 1. Music Control Mode (V-Sign is active)
        if (isVSign) {
            // Track skipping swipe
            const triggerSwipeThreshold = 0.0025; // Speed threshold to register a V-swipe
            if (Math.abs(vx) > triggerSwipeThreshold) {
                const now = Date.now();
                if (now - lastTrackChangeTime > 1500) {
                    lastTrackChangeTime = now;
                    if (vx > 0) {
                        gestureStatus.textContent = '곡 넘김: 다음 곡 ▶▶ (V-지문)';
                        nextSong();
                    } else {
                        gestureStatus.textContent = '곡 넘김: 이전 곡 ◀◀ (V-지문)';
                        prevSong();
                    }
                }
            } else {
                gestureStatus.textContent = 'V-제스처 감지됨: 좌우로 쓸어 곡 전환';
            }
        }
        // 2. Hologram Interactive Mode (Pointing finger or Fist grab, active ONLY when touching)
        else if (isTouchActive && (isPointingFinger || isFist)) {
            const sensitivity = dragSensitivity; // Bounded to UI sensitivity calibration slider
            hologramCube.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), vx * sensitivity);
            hologramCube.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), vy * sensitivity);

            // Compute velocities so when finger is released, the cube has natural inertia
            const maxInertia = 0.20; // Increased from 0.15 for faster flick capability
            rotVelY = Math.max(-maxInertia, Math.min(maxInertia, vx * sensitivity * 14.0));
            rotVelX = Math.max(-maxInertia, Math.min(maxInertia, vy * sensitivity * 14.0));
            rotVelZ = Math.max(-maxInertia * 0.3, Math.min(maxInertia * 0.3, (vx + vy) * sensitivity * 2.0));

            lastInteractionTime = Date.now();
            if (isFist) {
                gestureStatus.textContent = '홀로그램 움켜쥐기(그랩) 제어 중... ✊';
            } else {
                gestureStatus.textContent = '홀로그램 공간 터치 제어 중... 👆';
            }
        }
        else if (!isTouchActive && (isPointingFinger || isFist)) {
            gestureStatus.textContent = '호버 중: 손을 더 가까이 가져오세요 (가상 터치 활성화 대기)';
        }
    }

    // --- Gemini Voice Assistant Logic & Speech APIs ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        geminiRecognition = new SpeechRecognition();
        geminiRecognition.lang = 'ko-KR';
        geminiRecognition.continuous = false;
        geminiRecognition.interimResults = false;

        geminiRecognition.onstart = () => {
            console.log("[Gemini Speech] Listening started");
            gestureStatus.textContent = '제미나이: 듣고 있습니다... 🎙️';
        };

        geminiRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            console.log("[Gemini Speech] Recognized text:", transcript);
            gestureStatus.textContent = `제미나이 인식: "${transcript}"`;

            // Trigger thinking state and API call
            HologramStateManager.transitionTo('GEMINI', 'THINKING');
            sendToGemini(transcript);
        };

        geminiRecognition.onerror = (event) => {
            console.error("[Gemini Speech] Recognition error:", event.error);
            if (HologramStateManager.currentMode === 'GEMINI') {
                gestureStatus.textContent = '제미나이: 다시 말씀해 주세요... 🎙️';
                // Automatically restart listening after brief delay if still in gemini mode
                setTimeout(() => {
                    if (HologramStateManager.currentMode === 'GEMINI' && HologramStateManager.subState === 'LISTENING') {
                        try { geminiRecognition.start(); } catch (e) { }
                    }
                }, 1000);
            }
        };

        geminiRecognition.onend = () => {
            console.log("[Gemini Speech] Listening ended");
            // If still in gemini mode and listening, restart listening
            setTimeout(() => {
                if (HologramStateManager.currentMode === 'GEMINI' && HologramStateManager.subState === 'LISTENING') {
                    try { geminiRecognition.start(); } catch (e) { }
                }
            }, 800);
        };
    } else {
        console.warn("SpeechRecognition API not supported in this browser.");
    }

    function sendToGemini(text) {
        gestureStatus.textContent = '제미나이: 생각 중... ☄️';
        HologramStateManager.transitionTo('GEMINI', 'THINKING');

        fetch('/api/gemini', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: text })
        })
            .then(res => res.json())
            .then(data => {
                const answer = data.response;
                console.log("[Gemini Response]", answer);
                gestureStatus.textContent = `제미나이: "${answer}"`;

                // If a music command already transitioned us out of GEMINI mode, do not speak or transition to SPEAKING!
                if (HologramStateManager.currentMode !== 'GEMINI') {
                    console.log("[Gemini Response] Bypassing speech synthesis because mode is no longer GEMINI.");
                    return;
                }

                HologramStateManager.transitionTo('GEMINI', 'SPEAKING');
                speakGemini(answer);
            })
            .catch(err => {
                console.error("Gemini API error:", err);
                gestureStatus.textContent = '제미나이 API 호출 실패';

                // Return to listening
                if (HologramStateManager.currentMode === 'GEMINI') {
                    HologramStateManager.transitionTo('GEMINI', 'LISTENING');
                }
            });
    }

    let ttsAudio = null;
    let ttsPulseInterval = null;

    function stopTTS() {
        if (ttsPulseInterval) {
            clearInterval(ttsPulseInterval);
            ttsPulseInterval = null;
        }
        geminiScalePulse = 1.0;
        if (ttsAudio) {
            try { ttsAudio.pause(); } catch (e) {}
            ttsAudio = null;
        }
        if (window.speechSynthesis) {
            try { window.speechSynthesis.cancel(); } catch (e) {}
        }
    }

    function speakGemini(text) {
        if (!text) {
            if (HologramStateManager.currentMode === 'GEMINI') {
                HologramStateManager.transitionTo('GEMINI', 'LISTENING');
            }
            return;
        }

        const cleanText = text.replace(/\[ACTION:[^\]]+\]/g, '').trim();
        if (!cleanText) {
            if (HologramStateManager.currentMode === 'GEMINI') {
                HologramStateManager.transitionTo('GEMINI', 'LISTENING');
            }
            return;
        }

        stopTTS();

        // Use backend TTS MP3 endpoint (Reliable audio playback on Raspberry Pi / Linux Chrome)
        const ttsUrl = `/api/tts?text=${encodeURIComponent(cleanText)}`;
        ttsAudio = new Audio(ttsUrl);

        ttsAudio.onplay = () => {
            console.log("[Gemini TTS] Audio playback started via backend proxy");
            ttsPulseInterval = setInterval(() => {
                geminiScalePulse = (geminiScalePulse === 1.6) ? 1.0 : 1.6;
            }, 250);
        };

        ttsAudio.onended = () => {
            console.log("[Gemini TTS] Audio playback finished");
            stopTTS();
            if (HologramStateManager.currentMode === 'GEMINI') {
                HologramStateManager.transitionTo('GEMINI', 'LISTENING');
            }
        };

        ttsAudio.onerror = (e) => {
            console.warn("[Gemini TTS] Proxy audio failed, falling back to Web Speech API", e);
            stopTTS();
            speakGeminiWebSpeech(cleanText);
        };

        ttsAudio.play().catch(err => {
            console.warn("[Gemini TTS] Audio play error, falling back to Web Speech API", err);
            stopTTS();
            speakGeminiWebSpeech(cleanText);
        });
    }

    function speakGeminiWebSpeech(text) {
        if (!window.speechSynthesis) {
            if (HologramStateManager.currentMode === 'GEMINI') {
                HologramStateManager.transitionTo('GEMINI', 'LISTENING');
            }
            return;
        }

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        currentUtterance = utterance;
        utterance.lang = 'ko-KR';

        const voices = window.speechSynthesis.getVoices();
        const koVoice = voices.find(voice => voice.lang.includes('ko'));
        if (koVoice) {
            utterance.voice = koVoice;
        }

        utterance.onboundary = () => {
            geminiScalePulse = 1.6;
        };

        utterance.onend = () => {
            geminiScalePulse = 1.0;
            if (HologramStateManager.currentMode === 'GEMINI') {
                HologramStateManager.transitionTo('GEMINI', 'LISTENING');
            }
        };

        utterance.onerror = (e) => {
            console.error("[Gemini WebSpeech] Error:", e);
            geminiScalePulse = 1.0;
            if (HologramStateManager.currentMode === 'GEMINI') {
                HologramStateManager.transitionTo('GEMINI', 'LISTENING');
            }
        };

        window.speechSynthesis.speak(utterance);
    }

    // Explicitly load voices
    if (window.speechSynthesis) {
        window.speechSynthesis.getVoices();
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
        }
    }

    function toggleGeminiMode() {
        if (HologramStateManager.currentMode === 'GEMINI') {
            HologramStateManager.restorePreviousState();
        } else {
            wasPlayingBeforeGemini = isPlaying; // Backup playing state
            // Pause music if playing
            if (isPlaying) {
                pauseSong();
            }
            HologramStateManager.transitionTo('GEMINI', 'LISTENING');
        }
    }

    function drawHandLandmarks(landmarks) {
        // Draw joint skeleton on debug preview canvas
        debugCtx.fillStyle = '#ff3366';
        debugCtx.strokeStyle = '#00acff';
        debugCtx.lineWidth = 2;

        landmarks.forEach(point => {
            const px = (1.0 - point.x) * debugCanvas.width;
            const py = point.y * debugCanvas.height;
            debugCtx.beginPath();
            debugCtx.arc(px, py, 3, 0, Math.PI * 2);
            debugCtx.fill();
        });
    }

    // 8. Listen to socket.io events from Raspberry Pi hardware backend
    socket.on('gemini_toggle', () => {
        console.log("Hardware Gemini Toggle received!");
        toggleGeminiMode();
    });

    socket.on('gesture_trigger', (data) => {
        console.log("Hardware Gesture Triggered:", data);
        const actionType = data.type;
        const velocity = data.value || 0;

        // If in Gemini mode and the incoming gesture is NOT a music control command, ignore it to prevent visual glitches.
        // If it IS a music control command, we bypass Gemini immediately and switch to music mode!
        const isMusicCommand = (actionType === 'play' || actionType === 'pause' || actionType === 'next' || actionType === 'prev' || actionType === 'play_pause');
        if (HologramStateManager.currentMode === 'GEMINI') {
            if (!isMusicCommand) {
                return; // Ignore swipes, rotates, seeks during active voice sessions
            }
            // For music commands, we transition to music mode instantly!
            if (actionType === 'play' || actionType === 'next' || actionType === 'prev') {
                wasPlayingBeforeGemini = true;
                HologramStateManager.transitionTo('MUSIC', 'PLAYING');
            } else if (actionType === 'pause') {
                wasPlayingBeforeGemini = false;
                HologramStateManager.transitionTo('MUSIC', 'PAUSED');
            }
        }

        if (actionType === 'swipe') {
            // Apply spin to hologram on all axes
            rotVelY = velocity;
            rotVelX = velocity * 0.5;
            rotVelZ = velocity * 0.3;
            lastInteractionTime = Date.now();
            gestureStatus.textContent = `HW 제스처: 스와이프 회전 (속도: ${velocity.toFixed(3)})`;
        } else if (actionType === 'rotate') {
            const dx = data.dx || 0;
            const dy = data.dy || 0;

            // Sync touch/grab states from hardware detector packet
            isTouchActive = data.touch !== undefined ? data.touch : true;
            isGrabActive = data.grab !== undefined ? data.grab : false;

            // Sync absolute cursor position if provided in hardware packet
            if (data.x !== undefined) handX = 1.0 - data.x;
            if (data.y !== undefined) handY = data.y;

            // Set flag to true temporarily so automatic spin doesn't conflict
            isGestureDragging = isTouchActive;

            // Clear dragging flag after a short timeout of inactive packets
            if (window.hwRotateTimeout) clearTimeout(window.hwRotateTimeout);
            window.hwRotateTimeout = setTimeout(() => {
                isGestureDragging = false;
                isTouchActive = false;
                isGrabActive = false;
            }, 180); // slightly longer buffer for wireless latency

            const sensitivity = dragSensitivity * 0.12;
            hologramCube.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), dx * sensitivity);
            hologramCube.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), dy * sensitivity);

            const maxInertia = 0.05;
            rotVelY = Math.max(-maxInertia, Math.min(maxInertia, dx * sensitivity * 0.30));
            rotVelX = Math.max(-maxInertia, Math.min(maxInertia, dy * sensitivity * 0.30));
            rotVelZ = Math.max(-maxInertia * 0.3, Math.min(maxInertia * 0.3, (dx + dy) * sensitivity * 0.10));

            lastInteractionTime = Date.now();
            if (isGrabActive) {
                gestureStatus.textContent = 'HW 제스처: 홀로그램 움켜쥐기(그랩) 제어 중... ✊';
            } else {
                gestureStatus.textContent = 'HW 제스처: 홀로그램 공간 터치 제어 중... 👆';
            }
        } else if (actionType === 'play') {
            playSong();
            gestureStatus.textContent = 'HW 제스처: 음악 재생';
        } else if (actionType === 'pause') {
            pauseSong();
            gestureStatus.textContent = 'HW 제스처: 음악 일시정지';
        } else if (actionType === 'next') {
            nextSong(true);
            gestureStatus.textContent = 'HW 제스처: 다음 곡 재생';
        } else if (actionType === 'prev') {
            prevSong(true);
            gestureStatus.textContent = 'HW 제스처: 이전 곡 재생';
        } else if (actionType === 'play_pause') {
            togglePlay();
            gestureStatus.textContent = 'HW 제스처: 재생/일시정지 토글';
        } else if (actionType === 'seek') {
            if (audio.duration) {
                audio.currentTime = velocity * audio.duration;
                gestureStatus.textContent = `HW 제스처: 곡 재생구간 이동 (${Math.round(velocity * 100)}%)`;
                syncPlayerState();
            }
        }
    });

    // Remote settings synchronization
    socket.on('remote_setting_trigger', (data) => {
        console.log("Remote Setting Trigger received:", data);
        if (data.viewMode !== undefined) {
            window.hologramViewMode = data.viewMode;
            console.log("Hologram View Mode switched to:", window.hologramViewMode);
            syncPlayerState();
        }
        if (data.singleScale !== undefined) {
            window.singleModeScale = parseFloat(data.singleScale);
        }
        if (data.singleOffsetX !== undefined) {
            window.singleModeOffsetX = parseFloat(data.singleOffsetX);
        }
        if (data.friction !== undefined) {
            frictionSlider.value = data.friction;
            friction = parseFloat(data.friction);
        }
        if (data.sensitivity !== undefined) {
            sensitivitySlider.value = data.sensitivity;
            sensitivityValue.textContent = Math.round(data.sensitivity);
            dragSensitivity = parseFloat(data.sensitivity);
        }
        if (data.showGuides !== undefined) {
            guideToggle.checked = data.showGuides;
            toggleGuideLines();
        }
        if (data.webcamEnabled !== undefined) {
            webcamToggle.checked = data.webcamEnabled;
            if (data.webcamEnabled) {
                initMediaPipe();
            } else {
                stopMediaPipe();
            }
        }
    });

    // Remote Gemini USB Mic Synchronization
    socket.on('gemini_mic_state', (data) => {
        if (data.state === 'THINKING') {
            gestureStatus.textContent = '제미나이: 생각 중... ☄️';
            HologramStateManager.transitionTo('GEMINI', 'THINKING');
        }
    });

    socket.on('gemini_mic_response', (data) => {
        const answer = data.response;
        console.log("[Gemini USB Mic Response]", answer);
        gestureStatus.textContent = `제미나이: "${answer}"`;
        HologramStateManager.transitionTo('GEMINI', 'SPEAKING');
        speakGemini(answer);
    });

    // Remote Gemini Chat synchronization
    socket.on('remote_chat_trigger', (data) => {
        console.log("Remote Chat Trigger received:", data.prompt);
        if (data.prompt) {
            sendToGemini(data.prompt);
        }
    });

    // Helper function to sync player state to remote dashboard
    function syncPlayerState() {
        if (!socket || !socket.connected) return;
        socket.emit('player_state_sync', {
            isPlaying: isPlaying,
            currentSongIndex: currentSongIndex,
            currentTime: audio.currentTime || 0,
            duration: audio.duration || 0,
            trackTitle: trackTitle.textContent,
            trackArtist: trackArtist.textContent,
            trackCover: dashboardCover.src,
            interactionMode: interactionMode,
            hologramViewMode: window.hologramViewMode || 'QUAD',
            duration: audio.duration || 0,
            trackTitle: trackTitle.textContent,
            trackArtist: trackArtist.textContent,
            trackCover: dashboardCover.src,
            interactionMode: interactionMode,
            currentMode: HologramStateManager.currentMode,
            subState: HologramStateManager.subState,
            gestureStatus: gestureStatus.textContent
        });
    }

    // Periodic synchronization during active playback
    setInterval(() => {
        if (isPlaying) {
            syncPlayerState();
        }
    }, 1000);

    // Force a resize calculation on page load to ensure full viewport size is initialized correctly
    setTimeout(() => {
        onWindowResize();
    }, 150);
});
