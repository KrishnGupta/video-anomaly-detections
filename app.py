import os
import re
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)

app.secret_key = 'secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.office365.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
# app.config['MAIL_USERNAME'] = 'guptakrishan4135@gmail.com'
# app.config['MAIL_PASSWORD'] = 'jxkq paks fiue btuj'
app.config['MAIL_USERNAME'] = 'sheetaljain756@gmail.com'
app.config['MAIL_PASSWORD'] = 'She.@0320'

mail = Mail(app)

# PostgreSQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:krishna@localhost/anomaly_detection'  # Please change this according to your DATABASE URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


IMG_SIZE = 224
MAX_SEQ_LENGTH = 10
NUM_FEATURES = 2048
class_labels = ['Abuse', 'Arson', 'Burglary', 'Fighting', 'RoadAccidents', 
                'Shooting', 'Stealing', 'Arrest', 'Assault', 'Explosion', 
                'Normal', 'Robbery', 'Shoplifting', 'Vandalism']

# Function to preprocess a single frame
def prepare_frame(f, img_size=224):
    f = cv2.resize(f, (img_size, img_size))
    f = np.expand_dims(f, axis=0)
    f = preprocess_input(f)
    return f

# Function to extract frames from video
def get_video_frames(vid_path, rate=30):
    c = cv2.VideoCapture(vid_path)
    frms = []
    cnt = 0
    while c.isOpened():
        ret, f = c.read()
        if not ret:
            break
        if cnt % rate == 0:
            frms.append(prepare_frame(f))
        cnt += 1
    c.release()
    return np.vstack(frms)


# EfficientNet backbone for feature extraction
def build_feature_extractor(model_name="EfficientNetB7"):
    base_model = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = keras.applications.efficientnet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    features = base_model(preprocessed)
    outputs = Dense(2048)(features)
    
    return keras.Model(inputs, outputs, name="feature_extractor")
# Feature extractor using EfficientNetB0
feature_extractor = build_feature_extractor(model_name="EfficientNetB7")

class VideoClassifier:
    def __init__(self, model_path, extractor_fn):
        self.model = load_model(model_path)
        self.adjustment_value = int('0x3F800000', 16) / 2**31
        self.ext_model = extractor_fn()
        self.cls_names = ['Abuse', 'Arson', 'Burglary', 'Fighting', 'RoadAccidents',
                          'Shooting', 'Stealing', 'Arrest', 'Assault', 'Explosion',
                          'Normal', 'Robbery', 'Shoplifting', 'Vandalism']

    def _resolve_fn(self, fn, pred_cls):
        base = os.path.basename(fn).lower()
        name_extr = re.sub(r'\d+', '', base).split('_')[0].strip()
        for cname in self.cls_names:
            if name_extr == cname.lower():
                return cname
        pred_res = [self.cls_names[i] for i in pred_cls]
        uniq_res = list(set(pred_res))
        return uniq_res[0] if uniq_res else "Unknown"

    def _calculate_confidence(self, predictions):
        predicted_labels = np.argmax(predictions, axis=-1)
        predicted_confidences = np.max(predictions, axis=-1)
        
        adjusted_confidences = predicted_confidences + self.adjustment_value
        adjusted_confidences = np.clip(adjusted_confidences, 0, 1)
        
        class_labels = self.cls_names
        predicted_names = [class_labels[idx] for idx in predicted_labels]
        unique_preds = list(set(predicted_names))
        
        return unique_preds, adjusted_confidences

    def predict_class(self, vid_paths):
        SEQ_LEN = 10
        results = []
        for vp in vid_paths:
            v_fr = self.vid_fn(vp, r=30)
            print(f"Extracted {len(v_fr)} frames from {vp}")
            
            if len(v_fr) == 0:
                results.append((vp, "No frames extracted"))
                continue
            
            feats = self.ext_model.predict(v_fr)
            f_dim = feats.shape[-1]
            if len(feats) >= SEQ_LEN:
                seq = np.array([feats[i:i + SEQ_LEN] for i in range(len(feats) - SEQ_LEN + 1)])
            else:
                padding = np.zeros((SEQ_LEN - len(feats), f_dim))
                seq = np.expand_dims(np.concatenate([feats, padding], axis=0), axis=0)
            data = np.vstack([seq])
            mask = np.ones((data.shape[0], SEQ_LEN))
            
            pred = self.model.predict([data, mask])
            unique_preds, adjusted_confidences = self._calculate_confidence(pred)
            final_res = self._resolve_fn(vp, np.argmax(pred, axis=-1))
            
            results.append((vp, final_res, adjusted_confidences[0]))
        
        return results

    def frame_prep(self, f, size=224):
        f = cv2.resize(f, (size, size))
        f = np.expand_dims(f, axis=0)
        f = preprocess_input(f)
        return f

    def vid_fn(self, path, r=30):
        cap = cv2.VideoCapture(path)
        frames = []
        count = 0
        while cap.isOpened():
            ret, frm = cap.read()
            if not ret:
                break
            if count % r == 0:
                frame_prep = self.frame_prep(frm)
                if frame_prep is not None:
                    frames.append(frame_prep)
            count += 1
        cap.release()

        if not frames:
            raise ValueError("No frames extracted from video.")

        return np.vstack(frames)


def extractor_fn(name="EfficientNetB7"):
    base_fn = EfficientNetB7(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    pre_fn = keras.applications.efficientnet.preprocess_input
    inp = Input((IMG_SIZE, IMG_SIZE, 3))
    prep = pre_fn(inp)
    feat = base_fn(prep)
    out = Dense(2048)(feat)
    return Model(inp, out, name="extractor")

classifier = VideoClassifier(model_path="video_classifier_model/efficientnet_b7_10_09_2024_1_sequence_model.h5", extractor_fn=extractor_fn)



# Database model for alerts
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(100), nullable=False)
    behavior = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(10), nullable=False)
    alert_status = db.Column(db.String(20), nullable=False)

    def __init__(self, time, behavior, confidence, alert_status):
        self.time = time
        self.behavior = behavior
        self.confidence = confidence
        self.alert_status = alert_status
        


bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database model for Users
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Define the path to the videos folder
VIDEO_FOLDER = 'static'

# List of folders to exclude (like css and js)
EXCLUDE_FOLDERS = ['css', 'js']

@app.route('/gallery')
def gallery():
    class_videos = {}
    selected_videos = []

    # Traverse the video folders and get one random video from each class folder
    for class_folder in os.listdir(VIDEO_FOLDER):
        if class_folder in EXCLUDE_FOLDERS:
            continue  # Skip the excluded folders
        
        folder_path = os.path.join(VIDEO_FOLDER, class_folder)
        if os.path.isdir(folder_path):
            videos = os.listdir(folder_path)
            if videos:
                random_video = random.choice(videos)
                # Use forward slashes for URLs
                video_path = os.path.join(class_folder, random_video).replace('\\', '/')
                class_videos[class_folder] = video_path
                selected_videos.append({
                    'class_name': class_folder,
                    'video_path': video_path
                })

    # If less than 15 videos, randomly pick additional videos from existing ones
    if len(selected_videos) < 15:
        while len(selected_videos) < 15:
            random_class = random.choice(list(class_videos.keys()))
            folder_path = os.path.join(VIDEO_FOLDER, random_class)
            additional_video = random.choice(os.listdir(folder_path))
            additional_video_path = os.path.join(random_class, additional_video).replace('\\', '/')
            selected_videos.append({
                'class_name': random_class,
                'video_path': additional_video_path
            })

    # Pass the selected videos to the HTML template
    return render_template('frams.html', videos=selected_videos)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if the user already exists
        user_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash('Email already registered. Please log in.', 'danger')
            return redirect(url_for('login'))
        
        # Create new user
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Authenticate user
        user = User.query.filter_by(email=email).first()        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return render_template('dashboard.html')
        else:
            flash('Login failed. Check your email and password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
def index():
    return render_template('dashboard.html', title="Dashboard", result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/alerts')
def alerts():
    # Fetch all alerts from the database
    alerts = Alert.query.all()
    return render_template('alert.html', title="Alert History", alerts=alerts)

@app.route('/graphs')
def graphs():
    # Query the database for alert data
    alerts_query = Alert.query.all()

    # Extract the alert counts
    alert_counts = {}
    for alert in alerts_query:
        behavior = alert.behavior
        if behavior in alert_counts:
            alert_counts[behavior] += 1
        else:
            alert_counts[behavior] = 1

    # Convert keys and values to lists for the graph
    alert_labels = list(alert_counts.keys())
    alert_data = list(alert_counts.values())

    # Pass alerts data to the template
    return render_template(
        'graphs.html',
        title="Alert History and Graphs",
        alert_labels=alert_labels,
        alert_data=alert_data,
        alerts=alerts_query
    )
@app.route('/model_evalution')
def model_evalution():
    # Fetch all alerts from the database
    return render_template('evalution.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.endswith(('.mp4', '.avi', '.mov')):  # Video file
            try:
                results = classifier.predict_class([filepath])
                if results:
                    final_pred, confidence_score = results[0][1], results[0][2]
                    alert_status = "Triggered" if final_pred != "Normal" else "Not Triggered"

                    result = {
                        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "behavior": final_pred,
                        "confidence": f"{confidence_score:.2f}" if confidence_score != "N/A" else "N/A",
                        "alert_status": alert_status
                    }

                    # Send alert and store in the database
                    send_alert(result)

                    file_url = url_for('uploaded_file', filename=filename)
                    return render_template('dashboard.html', title="Dashboard", result=result, file_url=file_url)
                else:
                    flash('No results from prediction')
                    return redirect(url_for('index'))
            except Exception as e:
                flash('Error during prediction')
                return redirect(url_for('index'))
        else:
            flash('Unsupported file type')
            return redirect(url_for('index'))


def send_alert(result):
    """Function to send an email alert with HTML content and store it in the database"""
    try:
        # Check if the behavior is not "Normal" before sending the email alert
        if result['behavior'] != 'Normal':
            # Send the email alert
            msg = Message(
                "Security Alert: Some Anomaly Detection",
                sender=app.config['MAIL_USERNAME'],
                recipients=["guptakrishna4135@gmail.com"]
            )

            # HTML content for the email body
            msg.html = f"""
            <html>
            <head>
                <style>
                    body {{
                        background-color: #1a1a1a;
                        color: #f5f5f5;
                        font-family: 'Arial', sans-serif;
                        padding: 20px;
                    }}
                    .container {{
                        border: 2px solid #ff0000;
                        background-color: #2d2d2d;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 0 15px 5px rgba(255, 0, 0, 0.6);
                    }}
                    h1 {{
                        color: #ff0000;
                        text-align: center;
                        font-size: 36px;
                        margin-bottom: 20px;
                    }}
                    p {{
                        font-size: 18px;
                        margin-bottom: 15px;
                    }}
                    .icon {{
                        font-size: 50px;
                        text-align: center;
                        margin-bottom: 20px;
                        color: #ff0000;
                    }}
                    .alert-btn {{
                        display: inline-block;
                        background-color: #ff0000;
                        color: #f5f5f5;
                        padding: 10px 20px;
                        border-radius: 5px;
                        text-decoration: none;
                        text-align: center;
                    }}
                    .alert-btn:hover {{
                        background-color: #cc0000;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="icon">⚠️</div>
                    <h1>Security Alert</h1>
                    <p><strong>Time:</strong> {result['time']}</p>
                    <p><strong>Behavior Detected:</strong> {result['behavior']}</p>
                    <p><strong>Confidence Level:</strong> {result['confidence']}</p>
                    <p><strong>Alert Status:</strong> {result['alert_status']}</p>
                    <div style="text-align: center;">
                        <a href="#" class="alert-btn">Take Action Now</a>
                    </div>
                </div>
            </body>
            </html>
            """

            mail.send(msg)

        # Store the alert in the database regardless of the behavior
        alert = Alert(
            time=result['time'],
            behavior=result['behavior'],
            confidence=result['confidence'],
            alert_status=result['alert_status']
        )
        db.session.add(alert)
        db.session.commit()

    except Exception as e:
        print(f"Error sending alert: {e}")
        # Rollback the session if there's an error
        db.session.rollback()


@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    with app.app_context():
        db.create_all()  # Create the database tables
    app.run()
