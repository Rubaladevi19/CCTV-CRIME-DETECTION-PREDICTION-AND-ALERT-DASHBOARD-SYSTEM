Python 3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from flask import Flask, render_template, request, url_for, send_from_directory, redirect, Response
from collections import Counter
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
import datetime
import google.generativeai as genai
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Folder configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
REPORT_FOLDER = 'reports'
OUTPUT_FOLDER = 'static/outputs'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, REPORT_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load YOLO model
model_path = "C:\\Users\\serop\\OneDrive\\Desktop\\CRIME_ACTIVITY_PROJECT\\CRIME_ACTIVITY\\best.pt"
model = YOLO(model_path)

# Email configurations
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = 'daminmain@gmail.com'
EMAIL_PASSWORD = 'kpqtxqskedcykwjz'
EMAIL_TO = 'rubaladevi7@gmail.com'

def send_email_alert(activity_list, security_suggestion, report_filename, report_path):
    """Send an email alert with detection results and report attachment."""
    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = 'Suspicious Activity Detection Alert'

        # Email body
        body = f"""
        Suspicious Activity Detected!

        Detected Activities: {', '.join(activity_list)}
        
        Security Suggestion:
        {security_suggestion}

        A detailed report is attached for your reference.
        """
        msg.attach(MIMEText(body, 'plain'))

        # Attach the report file
        with open(report_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        # Encode the attachment
        encoders.encode_base64(part)

        # Add header to attachment
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {report_filename}'
        )

        # Add attachment to message
        msg.attach(part)

        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)

        # Send email
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_image(results, model, image_path):
    """Process image detection results and save annotated image."""
    output_path = os.path.join(OUTPUT_FOLDER, 'output_image.jpg')
    image = cv2.imread(image_path)
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        img = annotator.result()
        cv2.imwrite(output_path, img)
    return 'outputs/output_image.jpg'

def process_video(video_path, model):
    """Process video detection results and save annotated video."""
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    activity_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                activity_list.append(model.names[int(c)])

        annotated_frame = annotator.result()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return 'outputs/output_video.mp4', list(set(activity_list))

def run_object_detection(file_path, is_video=False):
    """Run object detection on image or video."""
    if is_video:
        output_path, activity_list = process_video(file_path, model)
        return activity_list, output_path
    else:
        results = model.predict(file_path)
        activity_counts = Counter(model.names[int(c)] for r in results for c in r.boxes.cls)
        activity_list = list(activity_counts.keys())
        output_path = process_image(results, model, file_path)
        return activity_list, output_path

def generate_suspicious_activity_suggestion(activity_list):
    """Generate security suggestions based on detected activities using Gemini AI."""
    prompt = (
        f"You are an expert in security and surveillance. Based on the following detected activities or objects: {', '.join(activity_list)}, "
        "provide a detailed analysis of potential suspicious behavior, identify risks, and suggest immediate actions to mitigate threats. "
        "Include actionable steps for security personnel to ensure safety."
    )
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.6,
                top_p=0.9
            )
        )
        suggestion = response.text.strip() if response.text else "No suggestion available."
    except Exception as e:
        suggestion = f"Error generating suggestion: {e}"
    return suggestion

def gen_frames():
    """Generate frames for live camera feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results[0].names[int(box.cls[0])]
            confidence = round(box.conf[0].item(), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def landing():
    """Render landing page."""
    return render_template('landing.html')

@app.route('/index')
def upload():
    """Render upload page."""
    return render_template('index.html')

@app.route('/live')
def live():
    """Render live camera page."""
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Stream live camera feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_file', methods=['POST'])
def process_file():
    """Process uploaded file and run detection."""
    if 'file' not in request.files:
        return render_template('result.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No selected file")

    allowed_image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    allowed_video_extensions = {'mp4', 'avi', 'mov'}
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    if file_ext not in allowed_image_extensions and file_ext not in allowed_video_extensions:
        return render_template('result.html', error="Invalid file type")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    is_video = file_ext in allowed_video_extensions
    activity_list, output_path = run_object_detection(file_path, is_video)
    if not output_path:
        return render_template('result.html', error="Error processing video")

    security_suggestion = generate_suspicious_activity_suggestion(activity_list)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{os.path.splitext(filename)[0]}_{timestamp}_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    with open(report_path, "w") as report_file:
        report_file.write(security_suggestion)

    # Send email alert after detection
    send_email_alert(activity_list, security_suggestion, report_filename, report_path)

    return render_template(
        'result.html',
        filename=filename,
        activity_list=activity_list,
        output_path=output_path,
        is_video=is_video,
        security_suggestion=security_suggestion,
        report_filename=report_filename
    )

@app.route('/static/outputs/<filename>')
def outputs(filename):
    """Serve output files."""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download report file."""
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)