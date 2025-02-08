from config_loader import ConfigLoader, setup_logging
import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import requests
from pathlib import Path
import logging
import time
from typing import Generator, Optional
import tempfile
import os

class ImageAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.setup_configs()
        self.setup_model()
        
    def setup_configs(self):
        """Initialize all configurations."""
        self.model_config = self.config_loader.get_model_config()
        self.face_config = self.config_loader.get_face_detection_config()
        self.emotion_config = self.config_loader.get_emotion_analysis_config()
        self.cache_config = self.config_loader.get_cache_config()
        self.api_config = self.config_loader.get_api_config()
        
        # Setup logging
        setup_logging(self.config_loader.get_logging_config())
        
        # Setup cache directory
        if self.cache_config.enabled:
            self.cache_dir = Path(self.cache_config.directory)
            self.cache_dir.mkdir(exist_ok=True)
            self.imagenet_cache = self.cache_dir / self.cache_config.imagenet_cache_file
    
    def setup_model(self):
        """Initialize the model based on configuration."""
        try:
            self.model = getattr(models, self.model_config.name)(
                pretrained=self.model_config.pretrained
            )
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize(self.model_config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.model_config.normalize['mean'],
                    std=self.model_config.normalize['std']
                ),
            ])
        except Exception as e:
            logging.error(f"Error setting up model: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        """Detect faces using configured method."""
        try:
            if self.face_config.detection_method == "haar_cascade":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.face_config.scale_factor,
                    minNeighbors=self.face_config.min_neighbors,
                    minSize=self.face_config.min_size
                )
                return faces
            # Add other detection methods here
            else:
                raise ValueError(f"Unsupported detection method: {self.face_config.detection_method}")
        except Exception as e:
            logging.error(f"Error detecting faces: {str(e)}")
            return np.array([])

    def analyze_facial_expression(self, image: np.ndarray) -> tuple:
        """Analyze facial expressions using configured parameters."""
        try:
            analysis = DeepFace.analyze(
                image,
                actions=self.emotion_config.actions,
                enforce_detection=self.emotion_config.enforce_detection
            )
            
            dominant_emotion = analysis[0]['dominant_emotion']
            emotions = analysis[0]['emotion']
            
            # Filter enabled emotions
            emotions = {k: v for k, v in emotions.items() 
                      if k in self.emotion_config.enabled_emotions}
            
            return dominant_emotion, emotions
        except Exception as e:
            logging.error(f"Error analyzing facial expression: {str(e)}")
            return "Error", {}

class VideoAnalyzer(ImageAnalyzer):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self.frame_count = 0
        self.processed_frames = 0
        
    def process_video_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Process a single video frame."""
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Analyze emotions if faces are detected
            results = {
                'faces_detected': len(faces),
                'emotions': [],
                'frame_number': self.frame_count
            }
            
            # Draw rectangles and analyze emotions for each face
            frame_with_faces = frame.copy()
            for (x, y, w, h) in faces:
                # Draw rectangle
                cv2.rectangle(
                    frame_with_faces,
                    (x, y),
                    (x+w, y+h),
                    self.face_config.draw_color,
                    self.face_config.rectangle_thickness
                )
                
                # Extract face region and analyze emotion
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:  # Check if ROI is not empty
                    try:
                        dominant_emotion, emotions = self.analyze_facial_expression(face_roi)
                        if dominant_emotion != "Error":
                            results['emotions'].append({
                                'position': (x, y, w, h),
                                'dominant_emotion': dominant_emotion,
                                'emotions': emotions
                            })
                            
                            # Add emotion label above face rectangle
                            label = f"{dominant_emotion}: {emotions[dominant_emotion]:.1f}%"
                            cv2.putText(
                                frame_with_faces,
                                label,
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.face_config.draw_color,
                                2
                            )
                    except Exception as e:
                        logging.warning(f"Error analyzing face in frame {self.frame_count}: {str(e)}")
                        continue
            
            return frame_with_faces, results
        except Exception as e:
            logging.error(f"Error processing frame {self.frame_count}: {str(e)}")
            return frame, {'error': str(e)}

    def process_video_stream(self, video_path: str) -> Generator[tuple[np.ndarray, dict], None, None]:
        """Process video file frame by frame."""
        try:
            cap = cv2.VideoCapture(video_path)
            self.frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.frame_count += 1
                processed_frame, results = self.process_video_frame(frame)
                yield processed_frame, results
                
            cap.release()
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            yield None, {'error': str(e)}

def run():
    # Load UI configuration
    config_loader = ConfigLoader()
    ui_config = config_loader.get_ui_config()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title=ui_config.page_title,
        page_icon=ui_config.page_icon,
        layout=ui_config.layout
    )
    
    st.title(ui_config.page_title)
    
    # Initialize analyzer
    try:
        analyzer = VideoAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {str(e)}")
        return

    # Add media type selector
    media_type = st.radio("Select media type:", ["Image", "Video"])
    
    if media_type == "Image":
        # Image processing
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=ui_config.supported_formats
        )
        
        if uploaded_file:
            try:
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                # Process and display image
                start_time = time.time()
                
                image = Image.open(uploaded_file)
                image_cv2 = np.array(image.convert('RGB'))
                image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                
                # Process image
                with st.spinner("Analyzing image..."):
                    faces = analyzer.detect_faces(image_cv2)
                    dominant_emotion, emotions = analyzer.analyze_facial_expression(image_cv2)
                    
                # Display results
                with col2:
                    st.subheader("Analysis Results")
                    
                    if ui_config.display['show_processing_time']:
                        processing_time = time.time() - start_time
                        st.caption(f"Processing Time: {processing_time:.2f} seconds")
                    
                    if dominant_emotion != "Error":
                        confidence = emotions.get(dominant_emotion, 0) * 100
                        st.success(f"Detected Emotion: {dominant_emotion.capitalize()}")
                        
                        if ui_config.display['show_confidence_scores']:
                            st.progress(confidence / 100)
                            st.caption(f"Confidence: {confidence:.2f}%")
                    else:
                        st.warning("No faces detected for emotion analysis")
                    
                    # Face detection visualization
                    if len(faces) > 0:
                        image_with_faces = image_cv2.copy()
                        for (x, y, w, h) in faces:
                            cv2.rectangle(
                                image_with_faces,
                                (x, y),
                                (x+w, y+h),
                                analyzer.face_config.draw_color,
                                analyzer.face_config.rectangle_thickness
                            )
                        
                        image_with_faces = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
                        st.image(image_with_faces, caption="Detected Faces", use_column_width=True)
                        st.success(f"Number of faces detected: {len(faces)}")
                    else:
                        st.warning("No faces detected in the image")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logging.error(f"Error processing image: {str(e)}")
            
    else:  # Video processing
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov']
        )
        
        if uploaded_video:
            # Check file size
            if uploaded_video.size > ui_config.max_upload_size_mb * 1024 * 1024:
                st.error(f"File size exceeds maximum limit of {ui_config.max_upload_size_mb}MB")
                return
                
            try:
                # Save uploaded video to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
                
                # Create video display area
                video_placeholder = st.empty()
                results_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Process video
                all_results = []
                total_emotions = {}
                
                try:
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    for frame_idx, (processed_frame, results) in enumerate(
                        analyzer.process_video_stream(video_path)
                    ):
                        if processed_frame is None:
                            continue
                            
                        # Update progress
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        
                        # Display processed frame
                        video_placeholder.image(
                            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                            use_column_width=True
                        )
                        
                        # Aggregate results
                        all_results.append(results)
                        for emotion_data in results.get('emotions', []):
                            emotion = emotion_data['dominant_emotion']
                            total_emotions[emotion] = total_emotions.get(emotion, 0) + 1
                        
                        # Display running statistics
                        stats_text = f"""
                        Processed frames: {frame_idx + 1}/{total_frames}
                        Faces detected: {results['faces_detected']}
                        """
                        results_placeholder.text(stats_text)
                    
                    # Display final statistics
                    st.subheader("Video Analysis Summary")
                    
                    # Calculate and display emotion distribution
                    if total_emotions:
                        st.write("Emotion Distribution:")
                        total_detections = sum(total_emotions.values())
                        for emotion, count in total_emotions.items():
                            percentage = (count / total_detections) * 100
                            st.write(f"{emotion}: {percentage:.1f}%")
                    else:
                        st.warning("No faces detected in the video")
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logging.error(f"Error processing video: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(video_path)
                    except Exception as e:
                        logging.error(f"Error removing temporary file: {str(e)}")
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logging.error(f"Error in video processing: {str(e)}")

if __name__ == "__main__":
    run()