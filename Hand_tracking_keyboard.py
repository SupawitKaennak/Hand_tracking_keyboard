import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import time
import pygame

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Load sound for key press (Make sure to provide the jcorrect path to your sound file)
key_sound = pygame.mixer.Sound("./sounds/key_press_sound.wav")  # Replace with your sound file path

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the keyboard controller
keyboard = Controller()

# Keyboard layout (full QWERTY layout) in lowercase
keyboard_keys = [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    ["z", "x", "c", "v", "b", "n", "m"],
    [Key.space]  # Use Key.space for the spacebar key
]

# Last time a key was pressed
last_pressed_time = 0
key_press_interval = 0.5  # Minimum interval between key presses (in seconds)

# Variable to display the last pressed key
last_key_pressed = ""


def draw_keyboard(frame, keys, key_size=(60, 60), margin=10):
    """Draw the on-screen keyboard and return the key positions."""
    height, width, _ = frame.shape
    key_positions = []
    y_offset = height - len(keys) * (key_size[1] + margin) - margin  # Start at the bottom

    for row_idx, row in enumerate(keys):
        x_offset = (width - len(row) * (key_size[0] + margin)) // 2  # Center the row
        for col_idx, key in enumerate(row):
            x = x_offset + col_idx * (key_size[0] + margin)
            y = y_offset + row_idx * (key_size[1] + margin)
            x_end = x + key_size[0]
            y_end = y + key_size[1]
            cv2.rectangle(frame, (x, y), (x_end, y_end), (255, 255, 255), -1)
            cv2.putText(frame, str(key), (x + 15, y + 45),  # Display key as text
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            key_positions.append(((x, y), (x_end, y_end), key))
    return key_positions


def is_inside_key(x, y, key_position):
    """Check if a point (x, y) is inside a key."""
    (top_left, bottom_right, _) = key_position
    return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]


# Check if key is "space" or a regular key and press accordingly
def press_key(key):
    """Press the key using pynput."""
    global last_pressed_time, last_key_pressed
    current_time = time.time()

    # Ensure there is a minimum interval between key presses
    if current_time - last_pressed_time >= key_press_interval:
        last_pressed_time = current_time
        last_key_pressed = key  # Update the last key pressed
        if key == Key.space:
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        else:
            keyboard.press(key)
            keyboard.release(key)

        # Play key press sound
        key_sound.play()


# Set up webcam feed
cap = cv2.VideoCapture(0)

# Set custom resolution for the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

# Create a resizable window
cv2.namedWindow("Hand Tracking with Input Keyboard", cv2.WINDOW_NORMAL)

# Initialize hand tracking with default settings
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        result = hands.process(rgb_frame)

        # Draw hand landmarks and connections
        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Get hand status (Left or Right)
                hand_status = result.multi_handedness[idx].classification[0].label
                confidence = result.multi_handedness[idx].classification[0].score

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display hand status on the frame
                x_min = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                y_min = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                text = f"{hand_status} ({confidence:.2f})"
                cv2.putText(frame, text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Get the position of the fingertips
                fingertip_positions = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                ]

                # Draw the fingertip cursor as a point
                for fingertip in fingertip_positions:
                    x = int(fingertip.x * frame.shape[1])
                    y = int(fingertip.y * frame.shape[0])

                    # Draw a small circle at the fingertip
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # Yellow circle for fingertip

                    # Draw the keyboard
                    key_positions = draw_keyboard(frame, keyboard_keys)

                    # Check if fingertip position is inside any key
                    for key_position in key_positions:
                        if is_inside_key(x, y, key_position):
                            _, _, key = key_position
                            press_key(key)  # Press the corresponding key

        # Display the last pressed key
        cv2.putText(frame, f"Last Key Pressed: {last_key_pressed}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the output
        cv2.imshow("Hand Tracking with Input Keyboard", frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
