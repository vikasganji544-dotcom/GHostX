import cv2
import mediapipe as mp
import numpy as np
import time

def run_calibration():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Recorded samples for threshold derivation
    pinch_samples = []
    
    print("\n" + "="*60)
    print(" GESTUREX THRESHOLD CALIBRATOR")
    print("="*60)
    print(" CONTROLS:")
    print("  [SPACE] : Record current pinch distance reading")
    print("  [C]     : Clear recorded readings")
    print("  [ESC]   : Exit and print recommended thresholds")
    print("="*60 + "\n")

    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe RGB processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)

        # FPS calculation
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Heads-up display (HUD) banner
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps} | Samples: {len(pinch_samples)}", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        live_pinch_dist = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # When cv2.flip is applied, invert label to match physical hand
                raw_label = handedness.classification[0].label
                hand_label = "Left" if raw_label == "Right" else "Right"

                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )

                # Extract pixel coordinates
                # Landmark 4: Thumb Tip, Landmark 8: Index Tip, Landmark 12: Middle Tip
                pts = {}
                for idx in [4, 8, 12]:
                    lm = hand_landmarks.landmark[idx]
                    pts[idx] = (int(lm.x * w), int(lm.y * h))

                # Right hand: Measure Thumb-to-Index pinch (cursor/click/drag)
                if hand_label == "Right":
                    t_pt = pts[4]
                    i_pt = pts[8]
                    dist = float(np.hypot(t_pt[0] - i_pt[0], t_pt[1] - i_pt[1]))
                    live_pinch_dist = dist

                    # Draw distance vector line
                    color = (0, 255, 0) if dist < 30 else (0, 140, 255)
                    cv2.line(frame, t_pt, i_pt, color, 3)

                    # Midpoint circle + measurement text
                    mid = ((t_pt[0] + i_pt[0]) // 2, (t_pt[1] + i_pt[1]) // 2)
                    cv2.circle(frame, mid, 5, (255, 255, 255), -1)
                    cv2.putText(frame, f"Right Pinch: {dist:.1f} px", (t_pt[0] - 40, t_pt[1] - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # Dynamic bar meter on left side of screen
                    bar_h = int(np.clip(dist * 2.5, 10, 300))
                    cv2.rectangle(frame, (20, 420 - bar_h), (45, 420), color, -1)
                    cv2.rectangle(frame, (20, 120), (45, 420), (255, 255, 255), 2)
                    cv2.putText(frame, f"{int(dist)}px", (15, 440),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Left hand: Measure Index-to-Middle separation (peace sign / right-click)
                elif hand_label == "Left":
                    i_pt = pts[8]
                    m_pt = pts[12]
                    left_dist = float(np.hypot(i_pt[0] - m_pt[0], i_pt[1] - m_pt[1]))

                    cv2.line(frame, i_pt, m_pt, (255, 180, 0), 2)
                    cv2.putText(frame, f"Left Fingers: {left_dist:.1f} px", (i_pt[0] - 40, i_pt[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

        # Instructions overlay
        cv2.putText(frame, "Space: Save Pinch | C: Reset | ESC: Done", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("GestureX Live Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if live_pinch_dist is not None:
                pinch_samples.append(live_pinch_dist)
                print(f"[+] Sample recorded: {live_pinch_dist:.2f} px")
            else:
                print("[!] No right-hand pinch detected to sample.")
        elif key in (ord('c'), ord('C')):
            pinch_samples.clear()
            print("[*] Sample buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()

    # Compute suggested hysteresis parameters
    print("\n" + "="*60)
    print(" CALIBRATION SUMMARY & RECOMMENDED VALUES")
    print("="*60)
    if pinch_samples:
        avg_pinch = float(np.mean(pinch_samples))
        min_pinch = float(np.min(pinch_samples))
        max_pinch = float(np.max(pinch_samples))
        
        # Recommended Hysteresis Gap
        rec_enter = round(avg_pinch * 1.15, 1)
        rec_exit = round(rec_enter + 18.0, 1)

        print(f"Recorded Samples : {len(pinch_samples)}")
        print(f"Min Distance     : {min_pinch:.1f} px")
        print(f"Max Distance     : {max_pinch:.1f} px")
        print(f"Mean Distance    : {avg_pinch:.1f} px")
        print("-" * 60)
        print("Update your GestureEngine variables in main.py with:")
        print(f"  DRAG_ENTER_DIST = {rec_enter}  # Trigger pinch")
        print(f"  DRAG_EXIT_DIST  = {rec_exit}  # Release pinch")
    else:
        print("No samples were captured via [SPACE].")
        print("Default baseline recommendations:")
        print("  DRAG_ENTER_DIST = 26.0")
        print("  DRAG_EXIT_DIST  = 46.0")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_calibration()