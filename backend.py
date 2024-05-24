from flask import Flask, render_template, Response,jsonify,request
import cv2
import pickle
import numpy as np
import mediapipe as mp
app = Flask(__name__)

# Sample function to generate a video feed (replace with your actual implementation)
def generate_video_feed():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to bytes
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        # Yield the frame as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('login.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/register')
def register():
    return render_template('register.html')
@app.route('/Home')
def Home():
    return render_template('index.html')

@app.route('/video_call')
def video_call():
    return render_template('video_call.html')

# Route to serve the video feed
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8',
    35: '9', 36: ' '
}

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)
def generate_frames():
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            # Draw predicted text on the frame
            cv2.putText(frame, f'Prediction: {predicted_character}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Check for space key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.putText(frame, "Space key pressed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat')
def chat():
    return render_template('chat.html')


videos = {
    'A': [
        {'url': 'https://videos.sproutvideo.com/embed/4498d3bc1215e0cecd/9a452ee5a9920103', 'title': 'A Little bit'},
        {'url': 'https://videos.sproutvideo.com/embed/7098d3bc1215e7c5f9/801abc57c895381d', 'title': 'A lot'},
        {'url': 'https://videos.sproutvideo.com/embed/1198d3bc1215e7c898/ebddae4a56fda859', 'title': 'About'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bc1215e6c025/7ea06aabb6df6a56', 'title': 'Above'},
        {'url': 'https://videos.sproutvideo.com/embed/a798d3bc1215e6c72e/8252fad30c30e6e3', 'title': 'Air Conditioner'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bc1215e5c325/c9c10b3044704409', 'title': 'Accident'},
        {'url': 'https://videos.sproutvideo.com/embed/1198d3bc1215e5ca98/8703d43c0c72da1e', 'title': 'Across'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bc1215e4c225/2d95206614896db5', 'title': 'Act'},
        {'url': 'https://videos.sproutvideo.com/embed/ea98d3bc1215e4c463/6e7d7e7430713b20', 'title': 'Action'},
        {'url': 'https://videos.sproutvideo.com/embed/7998d3bc1215ebccf0/dd895e550e874c7f', 'title': 'Add'},
        {'url': 'https://videos.sproutvideo.com/embed/a798d3bc1215ebca2e/a52a5b2d20ca3983', 'title': 'Address'},
        {'url': 'https://videos.sproutvideo.com/embed/d398d3bc1215eacf5a/d91b2f2385058298', 'title': 'Adult'},
        {'url': 'https://videos.sproutvideo.com/embed/ea98d3bc1215eaca63/a833778e18200edf', 'title': 'Advertisement'},
        {'url': 'https://videos.sproutvideo.com/embed/4d98d3bc1216e0c0c4/01966820f3a6a46c', 'title': 'Afraid'},
        {'url': 'https://videos.sproutvideo.com/embed/7998d3bc1216e1c5f0/2396f70b14d4bcdc', 'title': 'After'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bc1216e1c425/1446f76483ab4364', 'title': 'Afternoon'}
    ],
    'B': [
        {'url': 'https://videos.sproutvideo.com/embed/ea98d3bf191ee0c363/2de8b6008078aeff', 'title': 'Baby'},
        {'url': 'https://videos.sproutvideo.com/embed/0698d3bf191ee1c68f/fbd7954d14a2cbb8', 'title': 'Back'},
        {'url': 'https://videos.sproutvideo.com/embed/7098d3bf191ee1c0f9/416e0f1413b99901', 'title': 'BackPack'},
        {'url': 'https://videos.sproutvideo.com/embed/4498d3bf191ee1cccd/096323fe32785b5f', 'title': 'Bad'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bf191ee3c625/1fdbea956df99efe', 'title': 'Ball'},
        {'url': 'https://videos.sproutvideo.com/embed/d398d3bf191ee3c55a/affbfe668a27fd1c', 'title': 'Balloon'},
        {'url': 'https://videos.sproutvideo.com/embed/4d98d3bf191ee3c3c4/e3b1c9085b3019af', 'title': 'Banana'},
        {'url': 'https://videos.sproutvideo.com/embed/a798d3bf191ee4c62e/e4b5dc12eb3f68dd', 'title': 'Baseball'},
        {'url': 'https://videos.sproutvideo.com/embed/ea98d3bf191ee6c563/f56b500959e6b111', 'title': 'Bathroom'},
        {'url': 'https://videos.sproutvideo.com/embed/7998d3bf191ee7c3f0/01ad0f6e6b35e226', 'title': 'Beach'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bf191ee8cd25/564a0ed953b215e3', 'title': 'Beautiful'},
        {'url': 'https://videos.sproutvideo.com/embed/ac98d3bf191fe1c525/8c454f9938caa462', 'title': 'Before'},
        {'url': 'https://videos.sproutvideo.com/embed/ea98d3bf191fe5c763/2255026b1cd9f8af', 'title': 'Best'},
        {'url': 'https://videos.sproutvideo.com/embed/1198d3bf191fe4c998/625732e5866ca480', 'title': 'Big'},
        {'url': 'https://videos.sproutvideo.com/embed/1198d3bf191fe7ca98/65692172c7f64347', 'title': 'Black'},
        {'url': 'https://videos.sproutvideo.com/embed/0698d3bf191fe8ce8f/b906620f64a6db95', 'title': 'Blouse'}
    ],
    # Add more letters here following the same format
}

@app.route('/categories')
def categories():
    return render_template('categories.html')

@app.route('/videos/<letter>')
def get_videos(letter):
    return jsonify(videos.get(letter.upper(), []))


videos2 = [
    {'letter': 'A', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-591-1.mp4'},
    {'letter': 'B', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-592-1.mp4'},
    {'letter': 'C', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-593-1.mp4'},
    {'letter': 'D', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-594-1.mp4'},
    {'letter': 'E', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-595-1.mp4'},
    {'letter': 'F', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-596-1.mp4'},
    {'letter': 'G', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-597-1.mp4'},
    {'letter': 'H', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-598-1.mp4'},
    {'letter': 'I', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-599-1.mp4'},
    {'letter': 'J', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-600-1.mp4'},
    {'letter': 'K', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-601-1.mp4'},
    {'letter': 'L', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-602-1.mp4'},
    {'letter': 'M', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-603-1.mp4'},
    {'letter': 'N', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-604-1.mp4'},
    {'letter': 'O', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-605-1.mp4'},
    {'letter': 'P', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-606-1.mp4'},
    {'letter': 'Q', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-607-1.mp4'},
    {'letter': 'R', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-608-1.mp4'},
    {'letter': 'S', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-609-1.mp4'},
    {'letter': 'T', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-610-1.mp4'},
    {'letter': 'U', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-611-1.mp4'},
    {'letter': 'V', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-612-1.mp4'},
    {'letter': 'W', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-613-1.mp4'},
    {'letter': 'X', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-614-1.mp4'},
    {'letter': 'Y', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-615-1.mp4'},
    {'letter': 'Z', 'link': 'https://media.spreadthesign.com/video/mp4/13/alphabet-letter-616-1.mp4'}
]


@app.route("/FingerSpelling", methods=['GET', 'POST'])
def finger():
    selected_video = None
    if request.method == 'POST':
        selected_letter = request.form.get('selected_letter')
        selected_video = next((video for video in videos2 if video['letter'] == selected_letter), None)
    return render_template('FingerSpelling.html', videos2=videos2, selected_video=selected_video)

if __name__ == '__main__':
    app.run(debug=True)
