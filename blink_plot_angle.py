import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from FaceMeshModule import FaceMeshGenerator
from utils import DrawingUtils
import os
import time
import math



class BlinkCounterandEARPlot:
    """
    A class to detect and count eye blinks in a video using facial landmarks.
    
    This class processes video frames to detect faces, track eye movements,
    calculate Eye Aspect Ratio (EAR), plot EAR, and count blinks in real-time.
    """
    
    # Define facial landmark indices for eyes
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]  # Points for EAR calculation
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]  # Points for EAR calculation
    
    # Define colors for visualization
    COLORS = {
        'GREEN': {'hex': '#56f10d', 'bgr': (86, 241, 13)},
        'BLUE': {'hex': '#0329fc', 'bgr': (30, 46, 209)},
        'RED': {'hex': '#f70202', 'bgr': None}
    }

    def __init__(self, video_path, threshold, consec_frames, interval_threshold, save_video, output_filename=None):
        """
        Initialize the BlinkCounter with video and detection parameters.
        
        Args:
            video_path (str): Path to the input video file
            threshold (float): EAR threshold for blink detection
            consec_frames (int): Number of consecutive frames below threshold to count as a blink
            save_video (bool): Whether to save the processed video
            output_filename (str): Name of the video file if saving
        """
        # Initialize core parameters
        self.generator = FaceMeshGenerator()
        self.video_path = video_path
        self.EAR_THRESHOLD = threshold
        self.CONSEC_FRAMES = consec_frames
        
        self.interval_threshold = interval_threshold
        
        # Initialize video saving parameters
        self._init_video_saving(save_video, output_filename)
        
        # Initialize tracking variables
        self._init_tracking_and_command_variables()
        
        # Initialize plotting
        self._init_plot()
        
        # Initialize angle estimation
        self._init_angle_estimation()
    
    def _init_angle_estimation(self):
        """Initialize variables for angle estimation."""
        # Face coordination in real world (3D model)
        self.face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)
        
        # Landmark indices for angle estimation
        self.angle_landmark_indices = [1, 9, 57, 130, 287, 359]
        
        # Current angles
        self.pitch = 0
        self.yaw = 0
        self.roll = 0

    def _init_video_saving(self, save_video, output_filename):
        """Initialize video saving parameters and create output directory if needed."""
        self.save_video = save_video
        self.output_filename = output_filename
        self.out = None
        
        if self.save_video and self.output_filename:
            save_dir = "DATA/VIDEOS/OUTPUTS/Variasi Pengguna/" #+ folder1 + "/" + folder2 + "/"
            os.makedirs(save_dir, exist_ok=True)
            self.output_filename = os.path.join(save_dir, self.output_filename)

    def _init_tracking_and_command_variables(self):
        """Initialize variables used for tracking blinks and frame processing."""
        self.blink_counter = 0
        self.frame_counter = 0
        self.frame_number = 0
        
        self.ear_values = []
        self.saved_ear_values = []
        self.frame_numbers = []
        self.saved_frame_numbers = []
        self.max_frames = 400
        self.new_w = self.new_h = None
        self.input_fourcc = None  # Format video input (YUY2, MJPG, dll)

        # Add default y-axis limits
        self.default_ymin = 0.17  # Typical minimum EAR value
        self.default_ymax = 0.44  # Typical maximum EAR value

    def _init_plot(self):
        """Initialize the matplotlib plot for EAR visualization."""
        # Set up dark theme plot
        plt.style.use('default')
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(10, 5), dpi=200)
        self.canvas = FigureCanvas(self.fig)
        
        # Configure plot aesthetics
        self._configure_plot_aesthetics()
        
        self._init_plot_data()

        self.fig.canvas.draw()

    def _configure_plot_aesthetics(self):
        """Configure the aesthetic properties of the plot."""
        # Set background colors
        self.fig.patch.set_facecolor("#FFFFFF")
        self.ax.set_facecolor("#FFFFFF")
        
        # Configure axes with default limits initially
        self.ax.set_ylim(self.default_ymin, self.default_ymax)
        self.ax.set_xlim(0, self.max_frames)
        
        # Set labels and title
        self.ax.set_xlabel("Frame Number", color='black', fontsize=12)
        self.ax.set_ylabel("EAR", color='black', fontsize=12)
        self.ax.set_title("Real-Time Eye Aspect Ratio (EAR)", 
                         color='black', pad=10, fontsize=18, fontweight='bold')
        
        # Configure grid and spines
        self.ax.grid(True, color="#000000", linestyle='--', alpha=0.7)
        for spine in self.ax.spines.values():
            spine.set_color('black')
        
        # Configure ticks and legend
        self.ax.tick_params(colors='black', which='both')

    def _init_plot_data(self):
        """Initialize the plot data and curves."""
        self.x_vals = list(range(self.max_frames))
        self.y_vals = [0] * self.max_frames
        self.Y_vals = [self.EAR_THRESHOLD] * self.max_frames
        
        # Create curves with explicit labels
        self.EAR_curve, = self.ax.plot(
            self.x_vals, 
            self.y_vals,
            color=self.COLORS['GREEN']['hex'],
            label="Eye Aspect Ratio",
            linewidth=2
        )
        
        self.threshold_line, = self.ax.plot(
            self.x_vals,
            self.Y_vals,
            color=self.COLORS['RED']['hex'],
            label="Blink Threshold",
            linewidth=2,
            linestyle='--'
        )
        
        # Add legend 
        self.legend = self.ax.legend(
            handles=[self.EAR_curve, self.threshold_line],
            loc='upper right',
            fontsize=10,
            facecolor='white',
            edgecolor='black',
            labelcolor='black',
            framealpha=0.8,
            borderpad=1,
            handlelength=2
        )

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        """
        Calculate the eye aspect ratio (EAR) for given eye landmarks.
        
        The EAR is calculated using the formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
        where p1-p6 are specific points around the eye.
        
        Args:
            eye_landmarks (list): Indices of landmarks for one eye
            landmarks (list): List of all facial landmarks
        
        Returns:
            float: Calculated eye aspect ratio
        """
        A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - 
                          np.array(landmarks[eye_landmarks[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - 
                          np.array(landmarks[eye_landmarks[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - 
                          np.array(landmarks[eye_landmarks[3]]))
        return (A + B) / (2.0 * C)

    def rotation_matrix_to_angles(self, rotation_matrix):
        """
        Calculate Euler angles from rotation matrix.
        
        Args:
            rotation_matrix: A 3*3 rotation matrix
            
        Returns:
            numpy array: Angles in degrees for each axis (pitch, yaw, roll)
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                         rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

    def estimate_head_pose(self, face_landmarks, frame_width, frame_height):
        """
        Estimate head pose angles from facial landmarks.
        
        Args:
            face_landmarks: Dictionary of facial landmark coordinates
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            tuple: (pitch, yaw, roll) angles in degrees
        """
        face_coordination_in_image = []
        
        # face_landmarks is a dictionary with landmark ID as key and (x, y) tuple as value
        for idx in self.angle_landmark_indices:
            if idx in face_landmarks:
                x, y = face_landmarks[idx]
                face_coordination_in_image.append([x, y])
        
        if len(face_coordination_in_image) != 6:
            return self.pitch, self.yaw, self.roll
        
        face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
        
        # The camera matrix
        focal_length = 1 * frame_width
        cam_matrix = np.array([[focal_length, 0, frame_width / 2],
                               [0, focal_length, frame_height / 2],
                               [0, 0, 1]])
        
        # The Distance Matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Use solvePnP function to get rotation vector
        success, rotation_vec, transition_vec = cv.solvePnP(
            self.face_coordination_in_real_world, face_coordination_in_image,
            cam_matrix, dist_matrix)
        
        if success:
            # Use Rodrigues function to convert rotation vector to matrix
            rotation_matrix, jacobian = cv.Rodrigues(rotation_vec)
            
            result = self.rotation_matrix_to_angles(rotation_matrix)
            self.pitch, self.yaw, self.roll = result
        
        return self.pitch, self.yaw, self.roll

    def _update_plot(self, ear):
        """Update the plot with new EAR values."""
        if len(self.ear_values) > self.max_frames:
            self.ear_values.pop(0)
            self.frame_numbers.pop(0)
            
        color = self.COLORS['BLUE']['hex'] if ear < self.EAR_THRESHOLD else self.COLORS['GREEN']['hex']
        
        self.EAR_curve.set_xdata(self.frame_numbers)
        self.EAR_curve.set_ydata(self.ear_values)
        self.EAR_curve.set_color(color)
        
        self.threshold_line.set_xdata(self.frame_numbers)
        self.threshold_line.set_ydata([self.EAR_THRESHOLD] * len(self.frame_numbers))
        
        
        if len(self.frame_numbers) > 1:
            x_min = min(self.frame_numbers)
            x_max = max(self.frame_numbers)
            if x_min == x_max:
                # Add a small padding if min and max are the same
                x_min -= 0.5
                x_max += 0.5
            self.ax.set_xlim(x_min, x_max)
        else:
            # Default limits for initialization
            self.ax.set_xlim(0, self.max_frames)

        # Ensure the legend remains visible
        if self.legend not in self.ax.get_children():
            self.legend = self.ax.legend(
                handles=[self.EAR_curve, self.threshold_line],
                loc='upper right',
                fontsize=10,
                facecolor='black',
                edgecolor='white',
                labelcolor='white',
                framealpha=0.8,
                borderpad=1,
                handlelength=2
            )
        
        # Redraw with better quality
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.EAR_curve)
        self.ax.draw_artist(self.threshold_line)
        self.ax.draw_artist(self.legend)
        self.fig.canvas.flush_events()

    def process_frame(self, frame):
        """
        Process a single frame to detect and analyze eyes.
        
        Returns:
            tuple: Processed frame and EAR value
        """
        frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)
        
        if not face_landmarks:
            return frame, None, None
            
        # Calculate EAR
        right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
        left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
        ear = (right_ear + left_ear) / 2.0
        
        # Estimate head pose angles
        h, w, _ = frame.shape
        pitch, yaw, roll = self.estimate_head_pose(face_landmarks, w, h)
        
        # Determine visualization color
        color = self.COLORS['BLUE']['bgr'] if ear < self.EAR_THRESHOLD else self.COLORS['GREEN']['bgr']
        
        # Draw landmarks and update blink counter
        self._draw_frame_elements(frame, face_landmarks, color, pitch, yaw, roll)
        
        return frame, ear, (pitch, yaw, roll)

    def _draw_frame_elements(self, frame, landmarks, color, pitch=0, yaw=0, roll=0):
        """Draw eye landmarks, blink counter, and head pose angles on the frame."""
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Draw eye landmarks
        for eye in [self.RIGHT_EYE, self.LEFT_EYE]:
            for loc in eye:
                cv.circle(frame, (landmarks[loc]), 2, color, cv.FILLED)
        
        # Draw blink counter
        DrawingUtils.draw_text_with_bg(
            frame, f"Kedip: {self.blink_counter}", (0, 30),
            font_scale=1, thickness=3,
            bg_color=color, text_color=(0, 0, 0)
        )
                      
              
        cv.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)
        cv.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2)

        DrawingUtils.draw_text_with_bg(
            frame, f'pitch: {int(pitch)}', (w - 150, 30),
            font_scale=0.75, thickness=1,
            bg_color=color, text_color=(0, 0, 0)
        )

        DrawingUtils.draw_text_with_bg(
            frame, f'yaw: {int(yaw)}', (w - 150, 60),
            font_scale=0.75, thickness=1,
            bg_color=color, text_color=(0, 0, 0)
        )

        DrawingUtils.draw_text_with_bg(
            frame, f'roll: {int(roll)}', (w - 150, 90),
            font_scale=0.75, thickness=1,
            bg_color=color, text_color=(0, 0, 0)
        )
            
                
    def process_video(self):
        """Process the entire video and detect blinks."""
        try:
            cap = cv.VideoCapture(self.video_path, cv.CAP_DSHOW)

            # Ambil nilai FourCC
            fourcc_val = int(cap.get(cv.CAP_PROP_FOURCC))

            # Konversi integer ke string 4 karakter
            # Rumus: mengambil byte per byte dari integer
            fourcc_str = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

            print(f"Format kamera saat ini: {fourcc_str}")

            # Deteksi format YUY2 dan catat untuk konversi
            self.input_fourcc = fourcc_str
            if self.input_fourcc == 'YUY2':
                print("Mendeteksi format YUY2 - akan dikonversi ke BGR untuk output video")

            if not cap.isOpened():
                raise IOError(f"Failed to open video: {self.video_path}")

            self._process_video_frames(cap)

            # self._save_ear_values_to_txt()
            # self._save_plot_image()
            
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            if self.out:
                self.out.release()
            cv.destroyAllWindows()

    def _process_video_frames(self, cap):
        """Process individual frames from the video capture."""
        # Get video properties
        w =  int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h =  int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))

        # Validasi fps - jika 0 atau tidak valid, gunakan default 30
        if fps <= 0:
            print(f"FPS terdeteksi: {fps}, menggunakan default 30")
            fps = 30

        print(f"Video properties: {w}x{h} @ {fps} fps")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()

            frame = cv.flip(frame, 1)  # Mirror the frame for natural interaction

            # Debug: cek format frame
            if self.frame_number == 0:
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

            # Process frame and get EAR
            frame, ear, angles = self.process_frame(frame)
            
            if ear is not None:
                self._update_blink_detection(ear)
                
                pitch, yaw, roll = angles if angles else (0, 0, 0)        
                self._update_visualization(frame, ear, fps)

            # Menghitung sisa waktu agar pas dengan FPS
            elapsed = time.time() - start_time
            sleep_time = (1.0 / fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if cv.waitKey(1) & 0xFF == ord('p'):
                break

    def _update_blink_detection(self, ear):
        """Update blink detection based on EAR value."""
        self.ear_values.append(ear)
        self.frame_numbers.append(self.frame_number)
        self.saved_ear_values.append(ear)
        self.saved_frame_numbers.append(self.frame_number)
                      

        if ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
            
        else:
            
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_counter += 1
                
            self.frame_counter = 0

        self.frame_number += 1

            
    def _update_visualization(self, frame, ear, fps):
        """Update the visualization including the plot and video output."""
        self._update_plot(ear)
        
        # Convert plot to image and resize
        plot_img = self.plot_to_image()
        plot_img_resized = cv.resize(
            plot_img,
            (frame.shape[1], int(plot_img.shape[0] * frame.shape[1] / plot_img.shape[1]))
        )
        
        # Stack frames and handle video output
        stacked_frame = cv.vconcat([frame, plot_img_resized])
        self._handle_video_output(stacked_frame, fps)

    def _handle_video_output(self, stacked_frame, fps):
        """Handle video output, including saving and display."""
        # Validasi dan debug frame
        if self.frame_number == 0:
            print(f"Stacked frame: shape={stacked_frame.shape}, dtype={stacked_frame.dtype}")
        
        # Pastikan frame dalam format yang benar (BGR uint8)
        if stacked_frame.dtype != np.uint8:
            stacked_frame = stacked_frame.astype(np.uint8)
        
        # Pastikan hanya 3 channel (BGR)
        if len(stacked_frame.shape) == 2:
            # Grayscale - konversi ke BGR
            stacked_frame = cv.cvtColor(stacked_frame, cv.COLOR_GRAY2BGR)
        elif stacked_frame.shape[2] == 4:
            # RGBA - konversi ke BGR
            stacked_frame = cv.cvtColor(stacked_frame, cv.COLOR_RGBA2BGR)
        
        # Initialize video writer if needed
        if self.new_w is None:
            self.new_w = stacked_frame.shape[1]
            self.new_h = stacked_frame.shape[0]
            print(f"VideoWriter initialized: {self.new_w}x{self.new_h} @ {fps} fps")
            if self.save_video:
                self.out = cv.VideoWriter(
                    self.output_filename,
                    cv.VideoWriter_fourcc(*'XVID'),
                    fps,
                    (self.new_w, self.new_h)
                )
                # Verifikasi VideoWriter berhasil dibuka
                if not self.out.isOpened():
                    print("ERROR: VideoWriter gagal dibuka!")

        # Save frame if requested
        if self.save_video and self.out.isOpened():
            self.out.write(stacked_frame)

        # Display frame
        resizing_factor = 1 
        resized_shape = (
            int(resizing_factor * stacked_frame.shape[1]),
            int(resizing_factor * stacked_frame.shape[0])
        )
        stacked_frame_resized = cv.resize(stacked_frame, resized_shape)
        cv.imshow("folder1  + folder2  + file", stacked_frame_resized)

    def plot_to_image(self):
        """Convert the matplotlib plot to an OpenCV-compatible image."""
        self.canvas.draw()
        
        buffer = self.canvas.buffer_rgba()
        img_array = np.asarray(buffer)
        
        # img_array shape: (height, width, 4) - RGBA
        # Konversi RGBA ke BGR untuk OpenCV
        img_bgr = cv.cvtColor(img_array, cv.COLOR_RGBA2BGR)
        
        # Pastikan dtype uint8
        if img_bgr.dtype != np.uint8:
            img_bgr = img_bgr.astype(np.uint8)
        
        return img_bgr
        
    def _save_ear_values_to_txt(self):
        """Save the EAR values and corresponding frame numbers to a text file."""

        if not self.frame_numbers or not self.ear_values:
            return
        
        if self.save_video and self.output_filename:
            # Folder yang sama dengan file video output
            output_dir = os.path.dirname(self.output_filename)
            base_name = os.path.splitext(os.path.basename(self.output_filename))[0]
            
            os.makedirs(output_dir, exist_ok=True)
            txt_path = os.path.join(output_dir, f"{base_name}.txt")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("frame_number\tear\n")
                for frame_id, ear in zip(self.saved_frame_numbers, self.saved_ear_values):
                    f.write(f"{frame_id}\t{ear:.6f}\n")            

    def _save_plot_image(self):
        """Save the final EAR plot as an image file in the same directory as the output video."""

        if self.save_video and self.output_filename:
            output_dir = os.path.dirname(self.output_filename)
            base_name = os.path.splitext(os.path.basename(self.output_filename))[0]
            os.makedirs(output_dir, exist_ok=True)
            plot_image_path = os.path.join(output_dir, f"{base_name}_plot.png")

            self.EAR_curve.set_xdata(self.saved_frame_numbers)
            self.EAR_curve.set_ydata(self.saved_ear_values)
            
            self.threshold_line.set_xdata(self.saved_frame_numbers)
            self.threshold_line.set_ydata([self.EAR_THRESHOLD] * len(self.saved_frame_numbers))

            self.fig.savefig(plot_image_path, bbox_inches='tight', facecolor=self.fig.get_facecolor())
 

def _save_multiseries_plot(self):
        """Save a multi-series plot comparing EAR values across different videos."""
        # This method can be implemented to save a combined plot for multiple videos if needed
        pass    

if __name__ == "__main__":
    # Example usage


    # for folder1 in ["500 lumen", "800 lumen", "1000 lumen", "1200 lumen", "1300 lumen"]:
    #     for folder2 in ["50 cm", "100 cm", "150 cm", "200 cm"]: 
    #         for file in ["0", "10", "20", "30", "40", "50"]:
                input_video_path = 0#Kombinasi/" + folder1 + "/" + folder2 + "/" + file + ".mp4"
                blink_counter = BlinkCounterandEARPlot(
                    video_path=input_video_path,
                    threshold=0.25,
                    consec_frames=1,
                    interval_threshold=10,
                    save_video=True,
                    output_filename= "file.mp4" #file + ".mp4"
                )
                blink_counter.process_video()
                            
            

        