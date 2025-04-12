# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import subprocess
import threading
import queue
import time
import shutil # Keep for check_ffmpeg_gui
import cv2  # Requires opencv-python (and potentially opencv-contrib-python)
import numpy as np
from PIL import Image, ImageTk  # Requires Pillow: pip install Pillow
import signal  # For handling Ctrl+C
from multiprocessing import freeze_support # Still needed for bundling
import traceback  # For detailed error logging
# Removed re as it's no longer used for stderr parsing

# --- Optional Theme ---
try:
    # Attempt to import ttkthemes for enhanced styling
    from ttkthemes import ThemedTk
    THEMED = True
    DEFAULT_THEME = "clam" # A theme generally available and decent contrast
except ImportError:
    # Fallback if ttkthemes is not installed
    THEMED = False
    print("Optional package 'ttkthemes' not found. Using default Tkinter theme.")


# --- Constants ---
# Define output formats and their properties (codec parameters, file extensions)
OUTPUT_FORMATS = {
    "HEVC": {
        "suffix": "_hevc",
        "ext": ".mov",
        "ffmpeg_cpu": ["-c:v", "libx265", "-tag:v", "hvc1", "-preset", "medium", "-crf", "24",
                       "-pix_fmt", "yuva420p", "-x265-params", "alpha=1", "-c:a", "copy"],
        "ffmpeg_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p6", "-cq", "24",
                         "-pix_fmt", "p010le", "-c:a", "copy"], # HW HEVC often lacks reliable alpha
        "ffmpeg_qsv": ["-c:v", "hevc_qsv", "-load_plugin", "hevc_hw", "-preset", "medium", "-crf", "24",
                       "-pix_fmt", "p010le", "-c:a", "copy"], # HW HEVC often lacks reliable alpha
        "ffmpeg_amf": ["-c:v", "hevc_amf", "-quality", "balanced", "-rc", "cqp", "-qp_p", "24",
                       "-pix_fmt", "p010le", "-c:a", "copy"], # HW HEVC often lacks reliable alpha
        "description": "HEVC (H.265) in MOV. Alpha support problematic, especially with HW accel."
    },
    "ProRes": {
        "suffix": "_prores",
        "ext": ".mov",
        "ffmpeg_cpu": ["-c:v", "prores_ks", "-profile:v", "4", # Profile 4 = 4444 XQ (includes alpha)
                       "-pix_fmt", "yuva444p10le", "-c:a", "copy"],
        "description": "ProRes 4444 XQ in MOV. High-quality, large file, good alpha support (CPU only)."
    },
    "WebM": {
        "suffix": "_vp9",
        "ext": ".webm",
        "ffmpeg_cpu": ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva444p", # YUVA for alpha
                       "-crf", "30", "-b:v", "0", "-row-mt", "1", "-c:a",
                       "libopus", "-b:a", "128k", "-map_metadata", "-1"],
        "description": "VP9 in WebM (using yuva444p). Good compression, good alpha support (CPU only)."
    },
    "PNG": {
        "suffix": "_png",
        "ext": ".mov",
        "ffmpeg_cpu": ["-c:v", "png", "-pix_fmt", "rgba", "-c:a", "copy"], # RGBA native to PNG
        "description": "PNG codec in MOV. Lossless, large file, good alpha support (CPU only)."
    }
}

# Hardware acceleration options mapping to ffmpeg keys in OUTPUT_FORMATS
# This applies ONLY to the FFmpeg encoding stage.
HW_ACCEL_OPTIONS = {
    "None": "ffmpeg_cpu",
    "NVIDIA NVENC": "ffmpeg_nvenc",
    "Intel QSV": "ffmpeg_qsv",
    "AMD AMF": "ffmpeg_amf"
}

# Threshold for background detection sensitivity. Lower = more sensitive (less removed).
BG_THRESHOLD = 35


# --- Core Processing Logic ---

class FrameProcessorBase:
    """Base class for frame processing (CPU, CUDA, OpenCL). Pipes frames to FFmpeg."""

    def __init__(self, video_path, background_path, progress_queue):
        self.video_path = video_path
        self.background_path = background_path
        self.progress_queue = progress_queue
        self.cancel_event = threading.Event()
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0

    def request_cancel(self):
        """Signals the processing thread to cancel."""
        if not self.cancel_event.is_set():
            self.send_status("Cancellation requested...")
            self.cancel_event.set()

    # --- Communication with GUI via Queue ---
    def send_status(self, message):
        """Sends a status message to the GUI queue."""
        if self.progress_queue:
            self.progress_queue.put({'type': 'status', 'message': message})

    def send_progress(self, current, total):
        """Sends a progress update for frame reading/processing to the GUI queue."""
        if self.progress_queue:
            self.progress_queue.put({'type': 'progress_read', 'current': current, 'total': total})

    def send_preview(self, frame_bgra):
        """Sends a downscaled preview image (expects BGRA format) to the GUI."""
        if not self.progress_queue or frame_bgra is None:
            return
        try:
            preview_h = 90
            aspect_ratio = frame_bgra.shape[1] / frame_bgra.shape[0]
            preview_w = max(1, int(preview_h * aspect_ratio))
            preview_frame = cv2.resize(frame_bgra, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            preview_frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGRA2RGBA)
            img = Image.fromarray(preview_frame_rgb, 'RGBA')
            self.progress_queue.put({'type': 'preview', 'image': img})
        except Exception as e:
            self.send_status(f"Preview Error: {e}")
            print(f"Preview Error: {e}\n{traceback.format_exc()}")

    # --- Setup Methods ---
    def initialize_video(self):
        """Opens video, gets properties, reports to GUI. Returns True on success."""
        print(f"Initializing video: {os.path.basename(self.video_path)}")
        cap = None
        try:
            self.send_status(f"Opening video: {os.path.basename(self.video_path)}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Could not open video file: {self.video_path}")

            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.total_frames = int(frame_count_raw) if frame_count_raw and frame_count_raw > 0 else 0
            print(f"Video Properties Read: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS, RawFrameCount={frame_count_raw}, TotalFrames={self.total_frames}")

            # Basic validation of properties
            if self.frame_width <= 0 or self.frame_height <= 0 or self.fps <= 0:
                print("Warning: Invalid video properties reported by OpenCV.")
                # Try reading a frame to see if it works despite bad properties
                ret_test, _ = cap.read()
                if not ret_test:
                    raise RuntimeError(f"Invalid video properties AND failed to read first frame: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS")
                else:
                    # If frame read works, maybe properties were just wrong initially
                    self.send_status("Warning: Video properties may be inaccurate.")
                    # Re-fetch properties after reading a frame (sometimes helps)
                    self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.fps = cap.get(cv2.CAP_PROP_FPS)
                    if self.frame_width <= 0 or self.frame_height <= 0 or self.fps <= 0:
                         raise RuntimeError(f"Invalid video properties even after reading frame: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS")

            # Handle cases where frame count is unreliable
            if self.total_frames <= 0:
                self.send_status(f"Warning: Could not determine total frames accurately ({frame_count_raw}). Progress bars may be indeterminate.")
                self.total_frames = 0 # Use 0 to signify unknown total

            self.send_status(f"Video Info: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS, ~{self.total_frames or 'Unknown'} frames.")
            print("Video initialized successfully.")
            return True
        except Exception as e:
            self.send_status(f"Error initializing video: {e}")
            print(f"Error initializing video: {e}\n{traceback.format_exc()}")
            return False
        finally:
            # Ensure the capture object is released
            if cap:
                cap.release()
                # print("Video capture released in initialize_video finally block.") # Less verbose log

    def load_background(self):
        """Loads and resizes the background image (returns CPU version)."""
        self.send_status(f"Loading background: {os.path.basename(self.background_path)}")
        background = cv2.imread(self.background_path)
        if background is None:
            raise RuntimeError("Error: Could not load background image.")
        try:
            if self.frame_width <= 0 or self.frame_height <= 0:
                raise ValueError("Cannot resize background, invalid video dimensions obtained.")
            # Resize background to match video dimensions
            return cv2.resize(background, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            raise RuntimeError(f"Error resizing background: {e}")

    def run(self, ffmpeg_stdin):
        """Main processing logic - implemented by subclasses. Requires ffmpeg_stdin pipe."""
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    def _process_loop(self, cap, process_frame_func, ffmpeg_stdin):
        """
        Generic processing loop. Reads frames, processes them (to BGRA),
        converts to RGBA, and writes raw bytes to ffmpeg_stdin.
        """
        print("_process_loop started (piping RGBA mode).")
        frame_index = 0
        processed_frame_count = 0 # Count frames successfully piped
        last_preview_time = time.time()
        last_progress_update_time = time.time()

        try:
            while True:
                # Check for cancellation at the start of each loop
                if self.cancel_event.is_set():
                    self.send_status("Processing loop cancelled.")
                    break

                # Read the next frame
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video or error reading frame at index {frame_index}.")
                    break # Exit loop cleanly on end of video or read error

                try:
                    # Process the frame using the provided function (CPU/CUDA/OCL)
                    # This function is expected to return a BGRA numpy array
                    frame_bgra = process_frame_func(frame)
                    if frame_bgra is None:
                         self.send_status(f"Frame {frame_index} processing returned None. Skipping.")
                         frame_index += 1
                         continue # Skip to the next frame

                    # Send preview (using BGRA data) periodically
                    current_time = time.time()
                    if current_time - last_preview_time > 0.5: # Roughly 2 previews per second
                        self.send_preview(frame_bgra)
                        last_preview_time = current_time

                    # Convert frame to RGBA format required by FFmpeg pipe input
                    try:
                        frame_rgba = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGBA)
                    except cv2.error as conv_e:
                         self.send_status(f"Error converting frame {frame_index} to RGBA: {conv_e}. Skipping.")
                         print(f"Error converting frame {frame_index} to RGBA: {conv_e}")
                         frame_index += 1
                         del frame_bgra
                         continue

                    # Pipe the raw RGBA bytes to FFmpeg's standard input
                    try:
                        ffmpeg_stdin.write(frame_rgba.tobytes())
                        processed_frame_count += 1 # Increment only on successful write
                    except BrokenPipeError:
                        # This usually means FFmpeg closed its input pipe unexpectedly
                        self.send_status("Error: FFmpeg pipe broke. Stopping processing.")
                        print("ERROR: BrokenPipeError writing to FFmpeg stdin.")
                        self.request_cancel() # Signal cancellation to stop everything
                        break # Exit loop
                    except Exception as pipe_e:
                        # Handle other potential errors writing to the pipe
                        self.send_status(f"Error writing to FFmpeg pipe: {pipe_e}")
                        print(f"Error writing to FFmpeg pipe: {pipe_e}\n{traceback.format_exc()}")
                        self.request_cancel()
                        break

                    # Update frame reading progress periodically
                    # current_time = time.time() # Already have current_time from preview check
                    if current_time - last_progress_update_time > 1.0 or (self.total_frames > 0 and frame_index % 30 == 0):
                        self.send_progress(frame_index + 1, self.total_frames) # +1 because index is 0-based
                        last_progress_update_time = current_time

                    # Increment frame index and clean up memory
                    frame_index += 1
                    del frame_bgra
                    del frame_rgba

                # Handle errors during frame processing itself
                except cv2.error as e:
                    self.send_status(f"OpenCV Error processing frame {frame_index}: {e}. Skipping.")
                    print(f"OpenCV Error processing frame {frame_index}: {e}")
                    frame_index += 1
                    continue # Try next frame
                except Exception as e:
                    self.send_status(f"Unexpected Error processing frame {frame_index}: {e}. Skipping.")
                    print(f"Unexpected Error processing frame {frame_index}:\n{traceback.format_exc()}")
                    frame_index += 1
                    continue # Try next frame

            # --- Loop Finished ---
            print(f"_process_loop finished reading frames. Frames processed and sent to pipe: {processed_frame_count}")
            # Send final progress update for frames read
            self.send_progress(frame_index, self.total_frames)
            self.send_status(f"Frame processing finished. Sent {processed_frame_count} frames to FFmpeg.")
            return processed_frame_count # Return count of frames successfully processed and piped

        except Exception as e:
            # Catch critical errors in the loop setup or unexpected issues
            self.send_status(f"Critical Error during processing loop: {e}")
            print(f"Critical Error during processing loop:\n{traceback.format_exc()}")
            self.request_cancel() # Signal cancellation
            return processed_frame_count # Return count achieved so far
        finally:
            # --- Cleanup for this loop ---
            print("_process_loop entering finally block.")
            # Ensure FFmpeg's stdin pipe is closed to signal end of input
            if ffmpeg_stdin:
                try:
                    if not ffmpeg_stdin.closed:
                        print("Closing FFmpeg stdin pipe...")
                        ffmpeg_stdin.close()
                        print("FFmpeg stdin pipe closed.")
                    else:
                        print("FFmpeg stdin pipe already closed.")
                except Exception as close_e:
                    print(f"Warning: Error closing FFmpeg stdin pipe: {close_e}")

            # Release the video capture object
            if cap and cap.isOpened():
                cap.release()
                print("Video capture released in _process_loop finally block.")
            print("_process_loop finished finally block.")


# --- Processor Implementations ---

class CPUFrameProcessor(FrameProcessorBase):
    """Processes frames using standard CPU OpenCV functions."""
    def run(self, ffmpeg_stdin):
        print("CPUFrameProcessor run() started.")
        self.send_status("Initializing CPU processing...")
        try:
            background = self.load_background()
        except RuntimeError as e:
            self.send_status(str(e)); return 0

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.send_status("Error opening video for CPU processing."); return 0
        self.send_status(f"Processing frames (CPU) and piping to FFmpeg...")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        def process_frame(frame): # Returns BGRA
            diff = cv2.absdiff(frame, background)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY)
            # Refined Morphology: Close holes, then remove noise
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
            alpha = cv2.dilate(alpha, kernel, iterations=1) # Optional expansion
            # Combine with original frame using the mask
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            del diff, gray, alpha # Memory cleanup
            return bgra

        processed_count = self._process_loop(cap, process_frame, ffmpeg_stdin)
        del background # Memory cleanup
        self.send_status(f"CPU Frame processing finished. Sent {processed_count} frames.")
        print("CPUFrameProcessor run() finished.")
        return processed_count

class CUDAFrameProcessor(FrameProcessorBase):
    """Processes frames using OpenCV's CUDA module."""
    def run(self, ffmpeg_stdin):
        print("CUDAFrameProcessor run() started.")
        self.send_status("Initializing CUDA processing...")
        try:
            # Check CUDA availability
            if not hasattr(cv2, 'cuda') or cv2.cuda.getCudaEnabledDeviceCount() == 0:
                raise RuntimeError("CUDA not available or no CUDA devices found.")
            cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())
            self.send_status(f"Using CUDA Device: {cv2.cuda.getDevice()}")
        except Exception as e:
            self.send_status(f"CUDA Check/Info Error: {e}"); return 0

        # Load background and upload to GPU
        try:
            background_cpu = self.load_background()
            gpu_background = cv2.cuda_GpuMat()
            gpu_background.upload(background_cpu)
            del background_cpu
        except RuntimeError as e:
            self.send_status(str(e)); return 0
        except cv2.error as e:
            self.send_status(f"CUDA Error uploading background: {e}"); return 0

        # Open video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.send_status("Error opening video for CUDA processing."); return 0
        self.send_status(f"Processing frames (CUDA) and piping to FFmpeg...")

        # Pre-allocate GPU matrices
        gpu_frame_bgr = cv2.cuda_GpuMat()
        gpu_frame_bgra = cv2.cuda_GpuMat()
        gpu_diff = cv2.cuda_GpuMat()
        gpu_gray = cv2.cuda_GpuMat()
        gpu_alpha_thresh = cv2.cuda_GpuMat()
        gpu_alpha_closed = cv2.cuda_GpuMat()
        gpu_alpha_opened = cv2.cuda_GpuMat()
        gpu_alpha_final = cv2.cuda_GpuMat()

        # Create CUDA morphology filters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        try:
            close_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel)
            morph_open_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, kernel)
            dilate_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel, iterations=1)
            stream = cv2.cuda_Stream() # Use a stream for potential async operations
        except cv2.error as e:
            self.send_status(f"Failed to create CUDA filters/stream: {e}"); return 0

        def process_frame(frame): # Returns BGRA (downloaded from GPU)
            try:
                gpu_frame_bgr.upload(frame, stream)
                # Generate mask on GPU
                cv2.cuda.absdiff(gpu_frame_bgr, gpu_background, gpu_diff, stream=stream)
                cv2.cuda.cvtColor(gpu_diff, cv2.COLOR_BGR2GRAY, gpu_gray, stream=stream)
                cv2.cuda.threshold(gpu_gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY, gpu_alpha_thresh, stream=stream)
                close_filter.apply(gpu_alpha_thresh, gpu_alpha_closed, stream=stream)
                morph_open_filter.apply(gpu_alpha_closed, gpu_alpha_opened, stream=stream)
                dilate_filter.apply(gpu_alpha_opened, gpu_alpha_final, stream=stream)
                # Combine mask with original frame on GPU
                gpu_frame_bgra = cv2.cuda.cvtColor(gpu_frame_bgr, cv2.COLOR_BGR2BGRA, stream=stream)
                channels = cv2.cuda.split(gpu_frame_bgra, stream=stream)
                channels[3] = gpu_alpha_final
                cv2.cuda.merge(channels, gpu_frame_bgra, stream=stream)
                # Download result to CPU
                result_bgra = gpu_frame_bgra.download(stream=stream)
                stream.waitForCompletion() # Ensure operations complete before returning
                return result_bgra
            except cv2.error as e:
                self.send_status(f"CUDA processing error frame: {e}")
                print(f"CUDA processing error frame: {e}")
                stream.waitForCompletion() # Wait even on error
                return None
            except Exception as e:
                self.send_status(f"Non-OpenCV CUDA error frame: {e}")
                print(f"Non-OpenCV CUDA error frame:\n{traceback.format_exc()}")
                stream.waitForCompletion()
                return None

        # Run the main loop
        processed_count = self._process_loop(cap, process_frame, ffmpeg_stdin)

        # Cleanup GPU resources
        del gpu_background, gpu_frame_bgr, gpu_frame_bgra, gpu_diff, gpu_gray, gpu_alpha_thresh
        del gpu_alpha_closed, gpu_alpha_opened, gpu_alpha_final
        del close_filter, morph_open_filter, dilate_filter, stream

        self.send_status(f"CUDA Frame processing finished. Sent {processed_count} frames.")
        print("CUDAFrameProcessor run() finished.")
        return processed_count

class OpenCLFrameProcessor(FrameProcessorBase):
    """Processes frames using OpenCV's OpenCL (UMat) capabilities."""
    def run(self, ffmpeg_stdin):
        print("OpenCLFrameProcessor run() started.")
        self.send_status("Initializing OpenCL/UMat processing...")
        use_ocl = False
        ocl_needs_disable = False

        # Attempt to enable OpenCL
        try:
            if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                print(f"Attempting to enable OpenCL. Current status before enable: {cv2.ocl.useOpenCL()}")
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    print("OpenCL enabled successfully for processing.")
                    self.send_status("OpenCL acceleration enabled.")
                    use_ocl = True
                    ocl_needs_disable = True
                else:
                    # This case handles potential issues where enabling fails silently
                    self.send_status("OpenCL available but failed to enable. Check driver/OpenCV state.")
                    print("ERROR: OpenCL available but cv2.ocl.useOpenCL() returned False after setting True.")
                    cv2.ocl.setUseOpenCL(False) # Ensure it's off
            else:
                self.send_status("OpenCL not available in this OpenCV build.")
                print("OpenCL not available in this OpenCV build.")
        except Exception as e:
            self.send_status(f"Error during OpenCL enable attempt: {e}")
            print(f"Error during OpenCL enable attempt: {e}")
            try: # Attempt to disable OCL if error occurred during enable
                 if hasattr(cv2, 'ocl'): cv2.ocl.setUseOpenCL(False)
            except: pass

        # Exit if OpenCL could not be enabled
        if not use_ocl:
            self.send_status("Proceeding with CPU fallback (OpenCL unavailable/failed).")
            print("OpenCL not used, exiting OpenCLFrameProcessor.run() and returning 0.")
            return 0 # Indicate failure to initialize

        processed_count = 0
        cap = None
        backgroundUMat = None
        kernelUMat = None
        try:
            # Load background and convert to UMat (on device)
            try:
                background_cpu = self.load_background()
                backgroundUMat = cv2.UMat(background_cpu)
                del background_cpu
            except RuntimeError as e:
                self.send_status(str(e)); raise
            except cv2.error as e:
                self.send_status(f"OpenCL Error creating background UMat: {e}"); raise

            # Open video capture
            print("OpenCL Processor: Attempting to open VideoCapture...")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.send_status("Error opening video for OpenCL processing."); raise RuntimeError("Failed to open video capture")
            print("OpenCL Processor: VideoCapture opened successfully.")
            self.send_status(f"Processing frames (OpenCL) and piping to FFmpeg...")

            # Create kernel UMat
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernelUMat = cv2.UMat(kernel)

            def process_frame(frame): # Returns BGRA (downloaded from device)
                try:
                    # Upload frame to device (implicit UMat conversion)
                    frameUMat = cv2.UMat(frame)
                    # Generate mask using UMat operations (on device)
                    diffUMat = cv2.absdiff(frameUMat, backgroundUMat)
                    grayUMat = cv2.cvtColor(diffUMat, cv2.COLOR_BGR2GRAY)
                    _, alphaUMat_thresh = cv2.threshold(grayUMat, BG_THRESHOLD, 255, cv2.THRESH_BINARY)
                    alphaUMat_closed = cv2.morphologyEx(alphaUMat_thresh, cv2.MORPH_CLOSE, kernelUMat)
                    alphaUMat_opened = cv2.morphologyEx(alphaUMat_closed, cv2.MORPH_OPEN, kernelUMat)
                    alphaUMat_final = cv2.dilate(alphaUMat_opened, kernelUMat, iterations=1)
                    # Combine mask with original frame using UMat (on device)
                    frameUMat_bgra = cv2.cvtColor(frameUMat, cv2.COLOR_BGR2BGRA)
                    channels = list(cv2.split(frameUMat_bgra))
                    channels[3] = alphaUMat_final
                    cv2.merge(channels, frameUMat_bgra)
                    # Download result to CPU memory
                    result_bgra = frameUMat_bgra.get()
                    # Cleanup intermediate UMats for this frame
                    del frameUMat, frameUMat_bgra, diffUMat, grayUMat, alphaUMat_thresh
                    del alphaUMat_closed, alphaUMat_opened, alphaUMat_final, channels
                    return result_bgra # Return BGRA frame
                except cv2.error as e:
                    self.send_status(f"OpenCL processing error frame: {e}")
                    print(f"OpenCL processing error frame: {e}")
                    return None
                except Exception as e:
                    self.send_status(f"Non-OpenCV OpenCL error frame: {e}")
                    print(f"Non-OpenCV OpenCL error frame:\n{traceback.format_exc()}")
                    return None

            # Run the main loop
            processed_count = self._process_loop(cap, process_frame, ffmpeg_stdin)
            self.send_status(f"OpenCL Frame processing finished. Sent {processed_count} frames.")

        except Exception as e:
             # Handle errors during setup or loop initiation
             print(f"Error during OpenCL setup or process loop initiation: {e}")
             self.send_status(f"OpenCL Error: {e}")
             if processed_count == 0: processed_count = 0 # Ensure count is 0 if error before loop
             if not self.cancel_event.is_set(): self.request_cancel() # Trigger cancel on error
        finally:
            # --- Cleanup for OpenCL processor ---
            print("OpenCLFrameProcessor run() entering finally block.")
            # Release video capture if opened
            if cap and cap.isOpened(): cap.release(); print("OCL: VideoCapture released in finally.")
            # Explicitly delete UMats (helps GC)
            del backgroundUMat
            del kernelUMat

            # Disable OpenCL if it was successfully enabled
            if ocl_needs_disable:
                try:
                    print(f"Attempting to disable OpenCL. Current status before disable: {cv2.ocl.useOpenCL()}")
                    cv2.ocl.setUseOpenCL(False)
                    print(f"OpenCL disabled in finally block. Status after disable: {cv2.ocl.useOpenCL()}")
                    self.send_status("OpenCL disabled after processing.")
                except Exception as e:
                    print(f"Warning: Could not disable OpenCL in finally block: {e}")
            else:
                 print("OpenCL was not enabled, skipping disable step in finally block.")
            print("OpenCLFrameProcessor run() finished finally block.")

        return processed_count


# --- FFmpeg Handling ---
def check_ffmpeg_gui():
    """Checks for FFmpeg executable and returns path and warnings."""
    ffmpeg_path = shutil.which("ffmpeg")
    warnings = []
    if ffmpeg_path:
        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True,
                                    encoding='utf-8', errors='ignore', startupinfo=startupinfo, timeout=5)
            print(f"FFmpeg found at: {ffmpeg_path}")
            print(f"FFmpeg version info:\n{result.stdout.splitlines()[0]}")
        except FileNotFoundError:
            ffmpeg_path = None; warnings.append("Not found in PATH")
        except subprocess.TimeoutExpired:
            warnings.append("FFmpeg check timed out")
        except subprocess.CalledProcessError as e:
            warnings.append(f"FFmpeg check failed (Return Code: {e.returncode})")
        except Exception as e:
            warnings.append(f"Could not verify FFmpeg ({e})")
    else:
        warnings.append("Not found in PATH")
    return ffmpeg_path, warnings

def build_ffmpeg_pipe_command(width, height, fps, output_file, selected_format, audio_source_path, hw_accel_option):
    """Builds the FFmpeg command list for piping raw RGBA frames."""
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("Invalid video dimensions or FPS for FFmpeg command.")

    audio_source_path = os.path.abspath(audio_source_path)
    output_file = os.path.abspath(output_file)

    # Input options for raw RGBA video from stdin pipe
    cmd_input_pipe = ["-f", "rawvideo",
                      "-pixel_format", "rgba",
                      "-video_size", f"{width}x{height}",
                      "-framerate", str(fps),
                      "-i", "-"]

    # Base command: overwrite, report progress to stdout (pipe:1)
    # Remove -v verbose unless specifically needed for debugging other issues
    cmd_base = ["ffmpeg", "-y", "-progress", "pipe:1"]
    cmd_base.extend(cmd_input_pipe) # Add video pipe input
    cmd_base.extend(["-i", audio_source_path]) # Add audio file input

    # Get codec options based on user selection
    format_details = OUTPUT_FORMATS.get(selected_format)
    if not format_details: raise ValueError(f"Invalid format selection: {selected_format}")

    # Determine hardware acceleration parameters (or fallback to CPU)
    ffmpeg_key = HW_ACCEL_OPTIONS.get(hw_accel_option, "ffmpeg_cpu")
    if ffmpeg_key != "ffmpeg_cpu" and ffmpeg_key not in format_details:
        print(f"Warning: HWAccel '{hw_accel_option}' not defined for format '{selected_format}'. Falling back to CPU.")
        ffmpeg_key = "ffmpeg_cpu"

    cmd_codec = format_details.get(ffmpeg_key)
    if not cmd_codec: # Further fallback if specific key is missing
        print(f"Warning: Could not find ffmpeg parameters for key '{ffmpeg_key}'. Falling back to CPU.")
        cmd_codec = format_details.get("ffmpeg_cpu")
        if not cmd_codec: raise ValueError(f"Missing required ffmpeg CPU parameters for format: {selected_format}")

    # Map streams and define output file
    cmd_map_output = ["-map", "0:v:0",      # Map video from input 0 (pipe)
                      "-map", "1:a:0?",     # Map audio from input 1 (file), optional
                      output_file]

    # Combine all parts
    return cmd_base + cmd_codec + cmd_map_output

def ffmpeg_stderr_reader(process, progress_queue, cancel_event):
    """Reads FFmpeg stderr stream (binary), decodes, and sends LOG messages to the GUI queue."""
    print("Stderr reader: Started.")
    try:
        # Read line by line until stream closes
        for line_bytes in iter(process.stderr.readline, b''):
            if cancel_event.is_set(): break # Exit if cancelled
            # Decode and strip whitespace
            line_str = line_bytes.decode('utf-8', errors='replace').strip()
            if line_str: # Send non-empty lines as log messages
                progress_queue.put({'type': 'ffmpeg_log', 'message': line_str})
        print("Stderr reader: FFmpeg process stream ended.")
    except Exception as e:
        # Log errors during reading
        print(f"Error reading FFmpeg stderr: {e}")
        # Send error message to GUI log as well
        progress_queue.put({'type': 'ffmpeg_log', 'message': f'[Stderr reader error: {e}]'})
    finally:
        # Signal that this reader has finished
        progress_queue.put({'type': 'ffmpeg_log', 'message': '[Stderr reader finished]'})
        print("Stderr reader: Finished.")

def ffmpeg_stdout_reader(process, progress_queue, cancel_event):
    """Reads FFmpeg stdout stream (binary), decodes, and parses PROGRESS info."""
    print("Stdout reader: Started.")
    try:
        while True:
            # Check for cancellation
            if cancel_event.is_set():
                print("Stdout reader: Cancellation detected, stopping.")
                break

            # Read a line from stdout (where -progress pipe:1 writes)
            line_bytes = process.stdout.readline()
            if not line_bytes: # Break loop if FFmpeg closes the stdout pipe
                print("Stdout reader: FFmpeg process stream ended.")
                break

            # Decode the line
            line_str = line_bytes.decode('utf-8', errors='replace').strip()
            if not line_str: continue # Skip empty lines

            # Parse the 'key=value' format
            parts = line_str.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()

                # Look for the 'frame' key
                if key == 'frame':
                    try:
                        frame_num = int(value)
                        # Send progress update message
                        progress_queue.put({'type': 'progress_ffmpeg', 'current': frame_num})
                    except ValueError:
                        # Ignore if frame number isn't a valid integer
                        print(f"Stdout reader: Error parsing frame number: {value}")
                # Can add parsing for other keys like 'progress' or 'speed' if needed later
                # elif key == 'progress' and value == 'end':
                #    print("Stdout reader: Detected progress 'end'.")
                #    # Note: loop will also end when readline returns b''

        print("Stdout reader: Exiting loop.")

    except Exception as e:
        # Avoid logging errors if the process likely exited normally causing pipe closure
        if process.poll() is None: # Check if process is still running
            print(f"Error reading FFmpeg stdout: {e}")
            traceback.print_exc()
            progress_queue.put({'type': 'ffmpeg_log', 'message': f'[Stdout reader error: {e}]'})
        else:
            # Process already exited, likely pipe closed normally
            print(f"Stdout reader: Exited loop, process likely finished (return code: {process.returncode}).")
    finally:
        # Signal that this reader has finished
        progress_queue.put({'type': 'ffmpeg_log', 'message': '[Stdout reader finished]'})
        print("Stdout reader: Finished.")

def start_ffmpeg_pipe(ffmpeg_path, width, height, fps, output_file, selected_format,
                      audio_source_path, hw_accel_option, progress_queue, cancel_event):
    """
    Builds command, starts FFmpeg process with stdin pipe (binary mode),
    starts stderr AND stdout reader threads, returns process and threads.
    """
    process = None
    stderr_reader_thread = None
    stdout_reader_thread = None
    try:
        # Build the command, now including "-progress pipe:1"
        cmd = build_ffmpeg_pipe_command(width, height, fps, output_file, selected_format,
                                        audio_source_path, hw_accel_option)
        cmd[0] = ffmpeg_path # Ensure correct executable path
    except ValueError as e:
        progress_queue.put({'type': 'error', 'message': f"FFmpeg command build error: {e}"})
        return None, None, None # Indicate failure

    progress_queue.put(
        {'type': 'status', 'message': f"Starting FFmpeg for {selected_format} ({hw_accel_option})..."})
    print("--- Starting FFmpeg Process (Piping Mode with Progress to Stdout) ---")
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
    print("-------------------------------------------------------------------")

    try:
        # Configure startup info for Windows to hide console window
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        # Start FFmpeg process with pipes for stdin, stdout, stderr
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, # Capture stdout for progress
                                   stderr=subprocess.PIPE, # Capture stderr for logs
                                   startupinfo=startupinfo,
                                   bufsize=0) # Use default buffering for binary pipes

        # Start stderr reader thread (for logs)
        stderr_reader_thread = threading.Thread(target=ffmpeg_stderr_reader, args=(process, progress_queue, cancel_event),
                                         daemon=True)
        stderr_reader_thread.start()
        print("FFmpeg stderr reader thread launched.")

        # Start stdout reader thread (for progress)
        stdout_reader_thread = threading.Thread(target=ffmpeg_stdout_reader, args=(process, progress_queue, cancel_event),
                                         daemon=True)
        stdout_reader_thread.start()
        print("FFmpeg stdout reader thread launched.")

        # Return process handle and references to both reader threads
        return process, stderr_reader_thread, stdout_reader_thread

    except FileNotFoundError:
        progress_queue.put({'type': 'error', 'message': f"FFmpeg not found at path: {ffmpeg_path}"})
        return None, None, None
    except Exception as e:
        # Catch other potential errors during process startup
        progress_queue.put({'type': 'error', 'message': f"Unexpected FFmpeg startup Error: {e}"})
        traceback.print_exc()
        # Clean up if process started partially
        if process and process.poll() is None:
            try: process.kill(); process.wait()
            except: pass
        return None, None, None


# --- Tkinter GUI Application ---
class BackgroundRemoverApp:
    """Main class for the Tkinter GUI application."""

    def __init__(self, root):
        self.root = root
        self.root.title("Video Background Remover v15") # Updated title

        # --- Internal State ---
        self.video_path = tk.StringVar()
        self.background_path = tk.StringVar()
        self.output_file = tk.StringVar()
        self.selected_format = tk.StringVar(value=list(OUTPUT_FORMATS.keys())[0])
        self.hw_accel_selection = tk.StringVar(value=list(HW_ACCEL_OPTIONS.keys())[0])
        self.processing_mode = tk.StringVar(value="Detecting...")
        self.status_text = tk.StringVar(value="Ready.")
        self.ffmpeg_path = None
        self.ffmpeg_warnings = []

        # Batch processing state
        self.batch_mode_enabled = tk.BooleanVar(value=False)
        self.processing_queue = [] # List to hold job tuples: (vid, bg, out, fmt, hw)
        self.current_batch_index = -1 # Track which item in queue is processing

        self.progress_queue = queue.Queue() # Communication from threads to GUI
        self.processing_thread = None # The main background processing thread
        self.cancel_event = threading.Event() # Signals cancellation
        self.is_processing = False # Tracks if any processing (single or batch) is active
        self.preview_photo = None # Holds PhotoImage for preview label
        self.ffmpeg_log_buffer = [] # Buffer for log display (optional)
        self.ffmpeg_process = None # Reference to the current FFmpeg subprocess
        self.ffmpeg_stderr_reader_thread = None # Reference to stderr reader
        self.ffmpeg_stdout_reader_thread = None # Reference to stdout reader
        self.total_frames_for_progress = 0 # Total frames for current job

        # --- Initialization ---
        self.check_ffmpeg_on_startup()
        self.create_widgets()
        self.update_processing_mode()
        self.check_queue() # Start the GUI queue checker loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close

    def check_ffmpeg_on_startup(self):
        """Checks for FFmpeg when the app starts."""
        self.status_text.set("Checking FFmpeg...")
        self.ffmpeg_path, self.ffmpeg_warnings = check_ffmpeg_gui()
        status_msg = "Ready."
        if not self.ffmpeg_path:
            status_msg = "Error: FFmpeg not found in PATH!"
            messagebox.showerror("FFmpeg Error",
                             "FFmpeg executable not found in system PATH. Processing will fail. Please install FFmpeg and ensure it's accessible.")
        elif self.ffmpeg_warnings:
            warn_msg = f"FFmpeg Warning: {', '.join(self.ffmpeg_warnings)}"
            status_msg = warn_msg
            messagebox.showwarning("FFmpeg Warning", warn_msg + "\nSome operations might fail.")
        self.status_text.set(status_msg)

    def create_widgets(self):
        """Creates and lays out all the GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        # Configure column weights for responsiveness
        main_frame.columnconfigure(0, weight=1) # Let left column expand slightly
        main_frame.columnconfigure(1, weight=2) # Let right column (preview/log/queue) expand more

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Inputs / Job Configuration", padding="10")
        input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        self.video_entry = ttk.Entry(input_frame, textvariable=self.video_path, state='readonly', width=60)
        self.video_entry.grid(row=0, column=1, sticky="ew", pady=2)
        self.video_browse_btn = ttk.Button(input_frame, text="Browse...", command=self.browse_video)
        self.video_browse_btn.grid(row=0, column=2, sticky=tk.E, padx=5, pady=2)

        ttk.Label(input_frame, text="Background Img:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        self.bg_entry = ttk.Entry(input_frame, textvariable=self.background_path, state='readonly', width=60)
        self.bg_entry.grid(row=1, column=1, sticky="ew", pady=2)
        self.bg_browse_btn = ttk.Button(input_frame, text="Browse...", command=self.browse_background)
        self.bg_browse_btn.grid(row=1, column=2, sticky=tk.E, padx=5, pady=2)

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        output_frame.columnconfigure(1, weight=1)
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_file, state='readonly', width=60)
        self.output_entry.grid(row=0, column=1, sticky="ew", pady=2)
        self.output_select_btn = ttk.Button(output_frame, text="Select...", command=self.select_output)
        self.output_select_btn.grid(row=0, column=2, sticky=tk.E, padx=5, pady=2)

        ttk.Label(output_frame, text="Format:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=(0, 5))
        self.format_frame = ttk.Frame(output_frame) # Keep reference to disable/enable
        self.format_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W)
        for i, fmt in enumerate(OUTPUT_FORMATS.keys()):
            rb = ttk.Radiobutton(self.format_frame, text=fmt, variable=self.selected_format, value=fmt, command=self.auto_set_output)
            rb.grid(row=0, column=i, padx=5, sticky=tk.W)

        ttk.Label(output_frame, text="FFmpeg Accel:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=(0, 5))
        self.hw_accel_combo = ttk.Combobox(output_frame, textvariable=self.hw_accel_selection, values=list(HW_ACCEL_OPTIONS.keys()), state='readonly', width=15)
        self.hw_accel_combo.grid(row=2, column=1, sticky=tk.W, padx=5)

        # --- Processing Status Frame (Left Side) ---
        proc_frame = ttk.LabelFrame(main_frame, text="Processing Status", padding="10")
        proc_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        proc_frame.columnconfigure(0, weight=1)
        proc_frame.rowconfigure(5, weight=1) # Allow space below progress bars

        info_frame = ttk.Frame(proc_frame)
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(info_frame, text="Processing Mode:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(info_frame, textvariable=self.processing_mode, font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)

        self.progress1_label = ttk.Label(proc_frame, text="Frame Reading / Processing:")
        self.progress1_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.progress1 = ttk.Progressbar(proc_frame, orient=tk.HORIZONTAL, length=250, mode='determinate')
        self.progress1.grid(row=2, column=0, sticky="ew", pady=(0, 5))

        self.progress2_label = ttk.Label(proc_frame, text="FFmpeg Encoding:")
        self.progress2_label.grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.progress2 = ttk.Progressbar(proc_frame, orient=tk.HORIZONTAL, length=250, mode='determinate')
        self.progress2.grid(row=4, column=0, sticky="ew", pady=(0, 5))

        # Hide progress bars initially
        self.progress1_label.grid_remove()
        self.progress1.grid_remove()
        self.progress2_label.grid_remove()
        self.progress2.grid_remove()

        # --- Right Side Frame (Preview, Log, Queue) ---
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.rowconfigure(0, weight=1) # Preview area
        right_frame.rowconfigure(1, weight=1) # Log area
        right_frame.rowconfigure(2, weight=2) # Queue area (allow more space)
        right_frame.columnconfigure(0, weight=1)

        # Preview Area
        preview_frame = ttk.LabelFrame(right_frame, text="Preview", padding="5")
        preview_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, text="(Preview Area)", anchor=tk.CENTER, relief=tk.SUNKEN)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # Log Area
        log_frame = ttk.LabelFrame(right_frame, text="FFmpeg Log", padding="5")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(0,5))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=6, width=50, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1, font=("Consolas", 8) if os.name == 'nt' else ("Monaco", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")

        # --- Batch Queue Frame (Initially hidden potentially) ---
        self.queue_frame = ttk.LabelFrame(right_frame, text="Batch Queue", padding="10")
        # Don't grid initially, toggle_batch_mode will handle it
        self.queue_frame.columnconfigure(0, weight=1)
        self.queue_frame.rowconfigure(0, weight=1)

        # Queue Listbox
        self.queue_listbox = tk.Listbox(self.queue_frame, height=6, width=60)
        self.queue_listbox.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0, 5))
        queue_scrollbar = ttk.Scrollbar(self.queue_frame, orient=tk.VERTICAL, command=self.queue_listbox.yview)
        queue_scrollbar.grid(row=0, column=3, sticky="nsw", pady=(0,5))
        self.queue_listbox.config(yscrollcommand=queue_scrollbar.set)

        # Queue Buttons
        queue_button_frame = ttk.Frame(self.queue_frame)
        queue_button_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.add_queue_btn = ttk.Button(queue_button_frame, text="Add Current Job to Queue", command=self.add_to_queue)
        self.add_queue_btn.pack(side=tk.LEFT, padx=2)
        self.remove_queue_btn = ttk.Button(queue_button_frame, text="Remove Selected", command=self.remove_selected_from_queue)
        self.remove_queue_btn.pack(side=tk.LEFT, padx=2)
        self.clear_queue_btn = ttk.Button(queue_button_frame, text="Clear Queue", command=self.clear_queue)
        self.clear_queue_btn.pack(side=tk.LEFT, padx=2)

        # --- Bottom Controls (Batch Toggle & Start/Cancel) ---
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        bottom_frame.columnconfigure(0, weight=1) # Push buttons to the right

        # Batch Mode Toggle
        self.batch_toggle_cb = ttk.Checkbutton(bottom_frame, text="Enable Batch Mode",
                                               variable=self.batch_mode_enabled, command=self.toggle_batch_mode)
        self.batch_toggle_cb.grid(row=0, column=0, sticky="w", padx=5)

        # Action Buttons Frame
        button_frame = ttk.Frame(bottom_frame)
        button_frame.grid(row=0, column=1, sticky="e")
        button_style = 'Accent.TButton' if THEMED else 'TButton'
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing, style=button_style, width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED, width=10)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # --- Status Bar ---
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        status_bar.grid(row=1, column=0, sticky="ew")
        ttk.Label(status_bar, textvariable=self.status_text, anchor=tk.W).pack(fill=tk.X, padx=5)

        # Configure main frame row weights
        main_frame.rowconfigure(2, weight=1) # Allow processing/right frame row to expand

        # Initial UI state
        self.toggle_batch_mode() # Call once to set initial visibility

    # --- Batch Mode UI Toggle ---
    def toggle_batch_mode(self):
        """Shows or hides the batch queue UI elements based on the checkbox state."""
        if self.batch_mode_enabled.get():
            self.queue_frame.grid(row=2, column=0, sticky="nsew", pady=(5,0)) # Show queue frame
            self.start_button.config(text="Start Batch") # Change button text
        else:
            self.queue_frame.grid_remove() # Hide queue frame
            self.start_button.config(text="Start Processing") # Reset button text

    # --- Queue Management Functions ---
    def add_to_queue(self):
        """Adds the currently configured job to the processing queue."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot modify queue while processing.")
            return

        # Get current settings
        video_in = self.video_path.get()
        bg_in = self.background_path.get()
        file_out = self.output_file.get()
        fmt_out = self.selected_format.get()
        hw_accel = self.hw_accel_selection.get()

        # Validate inputs before adding
        if not all([video_in, bg_in, file_out, fmt_out]):
            messagebox.showerror("Input Error", "Please select input video, background image, output file, and format before adding to queue.")
            return
        if not os.path.isfile(video_in):
            messagebox.showerror("Input Error", f"Video file not found:\n{video_in}")
            return
        if not os.path.isfile(bg_in):
            messagebox.showerror("Input Error", f"Background image not found:\n{bg_in}")
            return
        # Basic check if output directory seems writable (doesn't guarantee success)
        output_dir = os.path.dirname(file_out)
        if not os.path.isdir(output_dir):
             try:
                 os.makedirs(output_dir, exist_ok=True) # Try creating output dir early
             except OSError as e:
                 messagebox.showerror("Output Error", f"Output directory does not exist and cannot be created:\n{output_dir}\nError: {e}")
                 return

        # Store job details as a tuple
        job_details = (video_in, bg_in, file_out, fmt_out, hw_accel)
        self.processing_queue.append(job_details)

        # Update listbox display (show only video filename for brevity)
        display_text = f"{os.path.basename(video_in)} -> {os.path.basename(file_out)} ({fmt_out}, {hw_accel})"
        self.queue_listbox.insert(tk.END, display_text)
        print(f"Added to queue: {display_text}")
        self.status_text.set(f"Job added to queue ({len(self.processing_queue)} total).")

    def remove_selected_from_queue(self):
        """Removes the selected job(s) from the queue."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot modify queue while processing.")
            return

        selected_indices = self.queue_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select a job from the queue to remove.")
            return

        # Remove items in reverse index order to avoid shifting issues
        for index in sorted(selected_indices, reverse=True):
            try:
                removed_item = self.processing_queue.pop(index)
                self.queue_listbox.delete(index)
                print(f"Removed item at index {index}: {os.path.basename(removed_item[0])}")
            except IndexError:
                print(f"Error removing item at index {index} - index out of bounds?")

        self.status_text.set(f"Selected job(s) removed ({len(self.processing_queue)} remaining).")

    def clear_queue(self):
        """Removes all jobs from the queue."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot modify queue while processing.")
            return

        if messagebox.askyesno("Confirm Clear", "Are you sure you want to remove all jobs from the queue?"):
            self.processing_queue.clear()
            self.queue_listbox.delete(0, tk.END)
            self.status_text.set("Queue cleared.")
            print("Queue cleared.")

    # --- Button Callbacks / File Dialogs ---
    def browse_video(self):
        path = filedialog.askopenfilename(title="Select Video File",
                                          filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                                                     ("All Files", "*.*")])
        if path:
            self.video_path.set(path)
            # Auto-suggest output only if not in batch mode (or maybe always?)
            # Let's auto-suggest always for convenience when adding jobs.
            self.auto_set_output()

    def browse_background(self):
        path = filedialog.askopenfilename(title="Select Background Image",
                                          filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                                                     ("All Files", "*.*")])
        if path:
            self.background_path.set(path)

    def select_output(self):
        video_in = self.video_path.get()
        initial_dir = os.path.dirname(self.output_file.get()) or os.path.dirname(video_in) or "."
        base = os.path.splitext(os.path.basename(video_in))[0] if video_in else "output"
        fmt = self.selected_format.get()
        format_details = OUTPUT_FORMATS.get(fmt)
        if not format_details: return
        suffix = format_details.get("suffix", "")
        default_ext = format_details.get("ext", ".mov")
        initial_file = base + "_processed" + suffix
        filetypes = [(f"{fmt} File (*{default_ext})", f"*{default_ext}"), ("All Files", "*.*")]
        path = filedialog.asksaveasfilename(title="Select Output File", initialdir=initial_dir,
                                            initialfile=initial_file, defaultextension=default_ext,
                                            filetypes=filetypes)
        if path: self.output_file.set(path)

    def auto_set_output(self):
        """Suggests an output filename based on input video and selected format."""
        video_in = self.video_path.get()
        if video_in:
            base = os.path.splitext(os.path.basename(video_in))[0]
            fmt = self.selected_format.get()
            format_details = OUTPUT_FORMATS.get(fmt)
            if not format_details: return
            suffix = format_details.get("suffix", "")
            default_ext = format_details.get("ext", ".mov")
            initial_file = base + "_processed" + suffix + default_ext
            output_dir = os.path.join(os.path.dirname(video_in) or ".", "BG_Removed_Output")
            self.output_file.set(os.path.join(output_dir, initial_file))
        elif not self.output_file.get():
             self.output_file.set("")

    # --- Processing Control ---
    def start_processing(self):
        """Starts processing either the single configured job or the batch queue."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return
        if not self.ffmpeg_path:
            messagebox.showerror("FFmpeg Error", "Cannot start: FFmpeg not found or not verified.")
            self.check_ffmpeg_on_startup()
            return

        is_batch = self.batch_mode_enabled.get()
        job_list = []

        if is_batch:
            if not self.processing_queue:
                messagebox.showerror("Queue Empty", "Batch mode is enabled, but the queue is empty. Add jobs to the queue first.")
                return
            # Process a copy of the queue list
            job_list = list(self.processing_queue)
            print(f"\n--- GUI: Initiating Start Batch Processing ({len(job_list)} jobs) ---")
        else:
            # Validate single job inputs
            video_in = self.video_path.get()
            bg_in = self.background_path.get()
            file_out = self.output_file.get()
            fmt_out = self.selected_format.get()
            hw_accel = self.hw_accel_selection.get()

            if not all([video_in, bg_in, file_out, fmt_out]):
                messagebox.showerror("Input Error", "Please select input video, background image, output file, and format.")
                return
            if not os.path.isfile(video_in): messagebox.showerror("Input Error", f"Video file not found:\n{video_in}"); return
            if not os.path.isfile(bg_in): messagebox.showerror("Input Error", f"Background image not found:\n{bg_in}"); return
            output_dir = os.path.dirname(file_out)
            try: os.makedirs(output_dir, exist_ok=True)
            except OSError as e: messagebox.showerror("Output Error", f"Could not create output directory:\n{output_dir}\nError: {e}"); return

            # Create a single-item list for the processing thread
            job_list = [(video_in, bg_in, file_out, fmt_out, hw_accel)]
            print("\n--- GUI: Initiating Start Single Job Processing ---")

        # --- Reset UI and State for New Run ---
        self.status_text.set("Starting...")
        self.progress1['value'] = 0
        self.progress1['maximum'] = 100
        self.progress1['mode'] = 'determinate'
        self.progress1_label.grid()
        self.progress1.grid()
        self.progress2['value'] = 0
        self.progress2['maximum'] = 100
        self.progress2['mode'] = 'determinate'
        self.progress2_label.grid()
        self.progress2.grid()

        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        # Disable queue modification buttons if in batch mode
        if is_batch:
            self.set_queue_buttons_state(tk.DISABLED)
            self.batch_toggle_cb.config(state=tk.DISABLED) # Disable toggle during batch run

        self.cancel_event.clear()
        self.is_processing = True
        self.total_frames_for_progress = 0 # Reset frame count
        self.ffmpeg_process = None
        self.ffmpeg_stderr_reader_thread = None
        self.ffmpeg_stdout_reader_thread = None
        self.clear_preview()
        self.clear_log()
        self.ffmpeg_log_buffer = []

        # --- Start Background Processing Thread ---
        print("GUI: Starting processing thread...")
        # Pass the list of jobs and the batch flag to the thread
        self.processing_thread = threading.Thread(target=self.run_processing_thread,
                                                  args=(job_list, is_batch),
                                                  daemon=True)
        self.processing_thread.start()

    def set_queue_buttons_state(self, state):
        """Helper to enable/disable queue management buttons."""
        if hasattr(self, 'add_queue_btn'): self.add_queue_btn.config(state=state)
        if hasattr(self, 'remove_queue_btn'): self.remove_queue_btn.config(state=state)
        if hasattr(self, 'clear_queue_btn'): self.clear_queue_btn.config(state=state)

    def cancel_processing(self):
        """Requests cancellation of the running process via the cancel_event."""
        if not self.is_processing or self.cancel_event.is_set(): return
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel the current process?\n(If in batch mode, the current job will stop, and the rest of the queue will be cancelled.)"):
            print("GUI: Cancellation confirmed by user.")
            self.cancel_event.set()
            self.status_text.set("Cancellation requested...")
            self.cancel_button.config(state=tk.DISABLED) # Disable cancel button after request

    def processing_finished(self, success=True):
        """Updates UI when processing thread signals completion (entire batch or single job)."""
        print(f"GUI: Processing finished callback received. Success: {success}, Cancelled: {self.cancel_event.is_set()}")
        # --- DEBUG ---
        print(f"GUI: processing_finished called. Current queue size: {len(self.processing_queue)}")
        # --- END DEBUG ---
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        # Re-enable queue buttons and toggle if batch mode was enabled
        if self.batch_mode_enabled.get():
            self.set_queue_buttons_state(tk.NORMAL)
            self.batch_toggle_cb.config(state=tk.NORMAL)

        # Hide progress bars
        self.progress1.grid_remove()
        self.progress1_label.grid_remove()
        self.progress2.grid_remove()
        self.progress2_label.grid_remove()
        self.progress1['value'] = 0
        self.progress2['value'] = 0

        # Determine final status message
        final_status = "Ready."
        if self.cancel_event.is_set():
            final_status = "Processing cancelled."
             # Update listbox to reflect remaining items after cancel
            if self.batch_mode_enabled.get():
                self.update_listbox_from_queue() # Refresh listbox
                final_status += f" {len(self.processing_queue)} jobs remaining."
        elif success:
            final_status = "Processing finished successfully."
            if self.batch_mode_enabled.get():
                if self.processing_queue: # Should be empty if successful batch finished
                     final_status = f"Batch finished with unexpected remaining jobs ({len(self.processing_queue)})."
                     self.update_listbox_from_queue() # Refresh listbox
                else:
                     final_status = "Batch finished successfully."

        else: # Processing failed
            if self.batch_mode_enabled.get():
                 # Update listbox based on potentially modified queue
                 self.update_listbox_from_queue()
                 final_status = f"Batch processing failed. {len(self.processing_queue)} jobs remaining."
            else:
                 final_status = "Processing failed. See status messages or console."
        self.status_text.set(final_status)

        self.ffmpeg_process = None # Clear process reference
        self.ffmpeg_stderr_reader_thread = None # Clear thread references
        self.ffmpeg_stdout_reader_thread = None
        self.current_batch_index = -1 # Reset batch index

        self.update_processing_mode() # Re-detect mode for next run
        print("GUI: Processing finished callback complete.")

    def update_listbox_from_queue(self):
        """Clears and repopulates the listbox based on self.processing_queue."""
        self.queue_listbox.delete(0, tk.END)
        for job in self.processing_queue:
            video_in, _, file_out, fmt_out, hw_accel = job
            display_text = f"{os.path.basename(video_in)} -> {os.path.basename(file_out)} ({fmt_out}, {hw_accel})"
            self.queue_listbox.insert(tk.END, display_text)
        print("GUI: Listbox updated from queue.")


    # --- Processing Thread Execution ---
    def run_processing_thread(self, job_list, is_batch):
        """
        The function executed by the background processing thread.
        Handles both single job and batch processing.
        """
        print(f"\n--- Processing Thread: Starting Run (Batch={is_batch}, Jobs={len(job_list)}) ---")
        overall_success = True
        # Use a copy for iteration, but modify the original self.processing_queue via index/message
        initial_queue_copy = list(job_list) # Keep track of original jobs for indexing display
        jobs_processed_count = 0 # Track how many jobs we actually finish processing

        for i, job_details in enumerate(initial_queue_copy):
            # Calculate the index in the *current* self.processing_queue
            # This assumes jobs are removed from the start upon success
            current_queue_index = i - jobs_processed_count
            self.current_batch_index = i # Track original index for status messages

            video_in, bg_in, file_out, fmt_out, hw_accel = job_details
            job_name = os.path.basename(video_in)
            print(f"\n--- Starting Job {i+1}/{len(initial_queue_copy)}: {job_name} ---")

            # Check for cancellation before starting next job
            if self.cancel_event.is_set():
                print(f"Cancellation detected before starting job {i+1}.")
                overall_success = False
                break # Exit the loop

            # Reset state for the new job
            self.ffmpeg_process = None
            self.ffmpeg_stderr_reader_thread = None
            self.ffmpeg_stdout_reader_thread = None
            self.total_frames_for_progress = 0
            job_success = False
            processor = None

            try:
                # --- Update GUI for current job ---
                status_prefix = f"Batch {i+1}/{len(initial_queue_copy)}: " if is_batch else ""
                self.progress_queue.put({'type': 'status', 'message': f"{status_prefix}Initializing {job_name}..."})
                # Send message to potentially highlight item in listbox (optional enhancement)
                # self.progress_queue.put({'type': 'highlight_queue_item', 'index': current_queue_index})

                # --- Phase 0: Setup and Video Info ---
                temp_processor = FrameProcessorBase(video_in, bg_in, self.progress_queue)
                if not temp_processor.initialize_video():
                    raise RuntimeError(f"Failed to initialize video properties for {job_name}.")
                width = temp_processor.frame_width
                height = temp_processor.frame_height
                fps = temp_processor.fps
                self.total_frames_for_progress = temp_processor.total_frames
                del temp_processor
                print(f"Job {i+1}: Total frames for progress bars: {self.total_frames_for_progress}")
                self.progress_queue.put({'type': 'set_total_frames', 'total': self.total_frames_for_progress})

                # --- Start FFmpeg Process ---
                self.ffmpeg_process, self.ffmpeg_stderr_reader_thread, self.ffmpeg_stdout_reader_thread = start_ffmpeg_pipe(
                    self.ffmpeg_path, width, height, fps, file_out, fmt_out,
                    video_in, hw_accel, self.progress_queue, self.cancel_event
                )
                if self.ffmpeg_process is None or self.ffmpeg_process.stdin is None:
                    raise RuntimeError(f"Failed to start FFmpeg process or pipe for {job_name}.")
                print(f"Job {i+1}: FFmpeg process started successfully.")

                # --- Phase 1: Instantiate Processor and Run Processing Loop ---
                mode = self.get_processing_mode() # Re-check mode for each job? Or assume constant? Let's re-check.
                self.progress_queue.put({'type': 'mode_update', 'mode': mode})
                self.progress_queue.put({'type': 'status', 'message': f"{status_prefix}Processing ({mode}) {job_name}..."})
                print(f"Job {i+1}: Instantiating processor for mode: {mode}")

                # Instantiate the correct processor
                if mode == "CUDA": processor = CUDAFrameProcessor(video_in, bg_in, self.progress_queue)
                elif mode == "OpenCL": processor = OpenCLFrameProcessor(video_in, bg_in, self.progress_queue)
                else: processor = CPUFrameProcessor(video_in, bg_in, self.progress_queue)

                # Set processor attributes
                processor.cancel_event = self.cancel_event
                processor.frame_width = width
                processor.frame_height = height
                processor.fps = fps
                processor.total_frames = self.total_frames_for_progress

                # Run the processing loop (pipes frames to FFmpeg's stdin)
                processed_frame_count = processor.run(self.ffmpeg_process.stdin)
                # processor.run() closes the stdin pipe in its finally block

                print(f"Job {i+1}: processor.run() completed. Sent {processed_frame_count} frames.")

                # Check for cancellation immediately after processing loop
                if self.cancel_event.is_set():
                    raise InterruptedError(f"Processing cancelled during frame processing for {job_name}.")

                # Validate frame count (important for non-CPU modes)
                if processed_frame_count <= 0 and mode != "CPU":
                    fail_reason = f"{mode} initialization likely failed."
                    raise RuntimeError(f"Processing ({mode}) failed for {job_name}: No frames processed/sent. {fail_reason}")
                elif processed_frame_count <= 0:
                    print(f"Warning: Job {i+1} ({job_name}) sent 0 frames to FFmpeg.")

                # --- Phase 2: Wait for FFmpeg to Finish ---
                self.progress_queue.put({'type': 'status', 'message': f"{status_prefix}Waiting for FFmpeg encoding ({job_name})..."})
                print(f"Job {i+1}: Waiting for FFmpeg process to finish...")
                # --- DEBUG ---
                print(f"Job {i+1}: Before ffmpeg_process.wait()")
                # --- END DEBUG ---
                ffmpeg_return_code = self.ffmpeg_process.wait() # Wait for FFmpeg process to exit
                # --- DEBUG ---
                print(f"Job {i+1}: After ffmpeg_process.wait() - Return Code: {ffmpeg_return_code}")
                # --- END DEBUG ---


                # --- Wait for reader threads for this specific job ---
                if self.ffmpeg_stderr_reader_thread and self.ffmpeg_stderr_reader_thread.is_alive():
                    print(f"Job {i+1}: Waiting for FFmpeg stderr reader thread (timeout 2s)...")
                    self.ffmpeg_stderr_reader_thread.join(timeout=2.0)
                    if self.ffmpeg_stderr_reader_thread.is_alive(): print(f"Job {i+1}: Warning: FFmpeg stderr reader thread still alive after join timeout.")
                    else: print(f"Job {i+1}: FFmpeg stderr reader thread joined.")


                if self.ffmpeg_stdout_reader_thread and self.ffmpeg_stdout_reader_thread.is_alive():
                    print(f"Job {i+1}: Waiting for FFmpeg stdout reader thread (timeout 2s)...")
                    self.ffmpeg_stdout_reader_thread.join(timeout=2.0)
                    if self.ffmpeg_stdout_reader_thread.is_alive(): print(f"Job {i+1}: Warning: FFmpeg stdout reader thread still alive after join timeout.")
                    else: print(f"Job {i+1}: FFmpeg stdout reader thread joined.")


                print(f"Job {i+1}: FFmpeg process finished with return code: {ffmpeg_return_code}")

                # Check FFmpeg success
                if ffmpeg_return_code != 0:
                    raise RuntimeError(f"FFmpeg encoding failed for {job_name} (Return Code: {ffmpeg_return_code}). Check log.")

                # If we reach here, this job was successful
                job_success = True
                jobs_processed_count += 1 # Increment successful job counter
                print(f"--- Job {i+1}/{len(initial_queue_copy)} ({job_name}) completed successfully ---")

                # If in batch mode, remove the completed job from the main queue list via GUI thread
                if is_batch:
                    # Send message to GUI to update the listbox using the *original* index 'i'
                    # The GUI needs to adjust based on how many items were already removed
                    self.progress_queue.put({'type': 'remove_queue_item_success', 'original_index': i})


            except InterruptedError as e:
                # Handle cancellation during this specific job
                self.progress_queue.put({'type': 'status', 'message': f"{status_prefix}{e}"})
                print(f"Processing Thread: Job {i+1} cancelled.")
                overall_success = False
                break # Stop processing further jobs in the batch
            except Exception as e:
                # Handle errors during this specific job
                err_msg =f"Error processing {job_name}: {e}"
                self.progress_queue.put({'type': 'status', 'message': f"{status_prefix}Error on {job_name}!"})
                self.progress_queue.put({'type': 'error', 'message': f"An error occurred processing job:\n{job_name}\n\nError: {e}\n\nCheck console for full traceback."})
                print(f"Error processing job {i+1} ({job_name}):\n{traceback.format_exc()}")
                overall_success = False
                # Stop batch processing on first error for simplicity
                break # Exit the loop
            finally:
                # --- Cleanup specific to this job ---
                print(f"Job {i+1}: Entering finally block.")
                # Ensure FFmpeg stdin is closed (should be done by _process_loop)
                if self.ffmpeg_process and self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                     print(f"Job {i+1} Finally: FFmpeg stdin pipe closing...")
                     try: self.ffmpeg_process.stdin.close()
                     except Exception as e: print(f"Error closing ffmpeg stdin in finally: {e}")

                # Ensure FFmpeg process is terminated if still running (e.g., due to cancel/error)
                if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    print(f"Job {i+1} Finally: FFmpeg process still running, terminating...")
                    try:
                        self.ffmpeg_process.terminate(); self.ffmpeg_process.wait(timeout=2.0)
                    except:
                        try: self.ffmpeg_process.kill(); self.ffmpeg_process.wait(timeout=1.0)
                        except: pass
                    print(f"Job {i+1} Finally: FFmpeg process terminated/killed.")

                # Ensure reader threads are joined (attempt again)
                if self.ffmpeg_stderr_reader_thread and self.ffmpeg_stderr_reader_thread.is_alive():
                     print(f"Job {i+1} Finally: Joining stderr reader...")
                     self.ffmpeg_stderr_reader_thread.join(timeout=0.5)
                if self.ffmpeg_stdout_reader_thread and self.ffmpeg_stdout_reader_thread.is_alive():
                     print(f"Job {i+1} Finally: Joining stdout reader...")
                     self.ffmpeg_stdout_reader_thread.join(timeout=0.5)

                # Reset progress bars for the next job (or final state) via queue
                # Use total=0 to reset/hide or make indeterminate
                self.progress_queue.put({'type': 'set_total_frames', 'total': 0})

                print(f"--- Finished Job {i+1}/{len(initial_queue_copy)} ---")

        # --- Loop Finished ---
        print("Processing thread finished job loop.")
        # Signal GUI that the entire process (single or batch) is finished
        # Success is true only if all jobs completed without error or cancellation
        final_success = overall_success and not self.cancel_event.is_set()
        # --- DEBUG ---
        print(f"Processing thread sending 'finished' message. Final success: {final_success}")
        # --- END DEBUG ---
        self.progress_queue.put({'type': 'finished', 'success': final_success})
        print("Processing thread finished.")


    def check_queue(self):
        """Checks the queue for messages and updates the GUI."""
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                msg_type = msg.get('type')

                # print(f"GUI Queue Received: {msg}") # Uncomment for heavy debugging

                if msg_type == 'status': self.status_text.set(msg['message'])
                elif msg_type == 'error':
                    err_short = msg['message'].splitlines()[0]
                    self.status_text.set(f"Error: {err_short}")
                    if not self.cancel_event.is_set(): messagebox.showerror("Processing Error", msg['message'])
                elif msg_type == 'set_total_frames':
                     total = msg.get('total', 0)
                     # print(f"GUI Update: Setting total frames for progress bars: {total}")
                     # Reset and configure progress bars for the current/next job
                     for p_bar in [self.progress1, self.progress2]:
                         if p_bar.winfo_exists():
                             p_bar.stop() # Stop indeterminate animation if running
                             p_bar['value'] = 0
                             if total > 0:
                                 p_bar['maximum'] = total
                                 p_bar['mode'] = 'determinate'
                             else: # Handle unknown total frames or reset
                                 p_bar['maximum'] = 1 # Avoid max=0 issues
                                 p_bar['mode'] = 'indeterminate'
                                 # Start pulsing only if processing is active and total is 0 (resetting between jobs)
                                 if self.is_processing and total <= 0:
                                     # Don't immediately restart if just finished
                                     pass
                                 elif self.is_processing:
                                      p_bar.start(50)
                     # Store total for percentage calculations
                     self.total_frames_for_progress = total if total > 0 else 0

                elif msg_type == 'progress_read':
                    current = msg.get('current', 0)
                    total = self.total_frames_for_progress
                    if self.progress1.winfo_exists():
                        if total > 0:
                            self.progress1['mode'] = 'determinate'
                            self.progress1['value'] = min(current, total) # Ensure value doesn't exceed max
                            percent = (current / total) * 100
                            # Update status only if processing normally
                            if self.is_processing and "Waiting" not in self.status_text.get():
                                status_prefix = f"Batch {self.current_batch_index+1}/{len(self.processing_queue)}: " if self.batch_mode_enabled.get() else ""
                                self.status_text.set(f"{status_prefix}Processing Frame: {current}/{total} ({percent:.1f}%)")
                        else: # Indeterminate mode
                            if not self.is_processing: self.progress1.stop()
                            elif self.progress1['mode'] == 'indeterminate': self.progress1.start(50) # Keep pulsing
                            if self.is_processing and "Waiting" not in self.status_text.get():
                                status_prefix = f"Batch {self.current_batch_index+1}/{len(self.processing_queue)}: " if self.batch_mode_enabled.get() else ""
                                self.status_text.set(f"{status_prefix}Processing Frame {current}...")

                elif msg_type == 'progress_ffmpeg':
                    current = msg.get('current', 0)
                    total = self.total_frames_for_progress
                    if self.progress2.winfo_exists():
                         if total > 0:
                             self.progress2['mode'] = 'determinate'
                             self.progress2['value'] = min(current, total) # Cap value at maximum
                         else: # Indeterminate mode
                             if not self.is_processing: self.progress2.stop()
                             elif self.progress2['mode'] == 'indeterminate': self.progress2.start(50) # Keep pulsing

                elif msg['type'] == 'preview':
                    # Update preview image if processing and widget exists
                    if msg.get('image') and self.is_processing and self.preview_label.winfo_exists():
                        try:
                            self.preview_photo = ImageTk.PhotoImage(msg['image'])
                            self.preview_label.config(image=self.preview_photo, text="")
                        except Exception as e: print(f"Error updating preview image: {e}"); self.preview_label.config(text="Preview Error", image='')
                elif msg_type == 'mode_update':
                    # Update the processing mode label (CPU/CUDA/OpenCL)
                    self.processing_mode.set(msg['mode'])
                elif msg['type'] == 'ffmpeg_log':
                    # Append FFmpeg log messages to the text widget
                    log_line = msg.get('message', '')
                    # Filter out progress lines from stdout if they somehow leak to log queue
                    if log_line and self.log_text.winfo_exists() and not log_line.startswith(('frame=', 'fps=', 'size=', 'time=', 'bitrate=', 'speed=', 'progress=', 'out_time_us=')):
                        self.log_text.config(state=tk.NORMAL)
                        self.log_text.insert(tk.END, log_line + '\n')
                        self.log_text.see(tk.END) # Auto-scroll
                        self.log_text.config(state=tk.DISABLED)
                elif msg_type == 'remove_queue_item_success':
                    # Remove the successfully completed item from the *live* queue data
                    # This now receives the *original* index from the iterated list
                    original_index = msg.get('original_index', -1)
                    # We need to find the item in the *current* queue that corresponds
                    # This is tricky if items could be identical. Let's assume for now
                    # removing the *first* item is correct after each success.
                    # A better approach would be to pass unique IDs.
                    # --- Simplified approach: Remove the first item ---
                    if self.processing_queue:
                        try:
                            removed_job_details = self.processing_queue.pop(0) # Remove from front
                            self.queue_listbox.delete(0) # Delete from top of listbox
                            print(f"GUI: Removed completed job {os.path.basename(removed_job_details[0])} from queue/listbox.")
                        except Exception as e:
                            print(f"GUI Error removing completed job from front: {e}")
                    else:
                         print(f"GUI Error: Tried to remove item from empty queue.")


                elif msg['type'] == 'finished':
                    # --- DEBUG ---
                    print("GUI: Received 'finished' message in check_queue.")
                    # --- END DEBUG ---
                    # Call the main finished handler
                    self.processing_finished(msg['success'])

        except queue.Empty: pass # No messages, just continue
        except Exception as e: print(f"Error in check_queue: {e}\n{traceback.format_exc()}")
        finally:
            # Reschedule the checker if the window still exists
            if self.root.winfo_exists():
                self.root.after(100, self.check_queue)

    def clear_log(self):
        """Clears the FFmpeg log text area."""
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
            self.ffmpeg_log_buffer = []

    def clear_preview(self):
        """Clears the preview image area."""
        if hasattr(self, 'preview_label') and self.preview_label.winfo_exists():
            self.preview_label.config(image='', text="(Preview Area)")
            self.preview_photo = None

    # --- Utility and Exit Handling ---
    def get_processing_mode(self):
        """Detects best available acceleration mode for OpenCV processing (CUDA > OpenCL > CPU)."""
        # This check is relatively quick and ensures the best mode is used for each job if run in batch
        print("Detecting Processing mode...")
        try: # Check CUDA
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    _gpu_mat_test = cv2.cuda_GpuMat(1, 1, cv2.CV_8U); del _gpu_mat_test
                    print("CUDA functional test passed.")
                    return "CUDA"
                except Exception as e: print(f"CUDA functional test failed: {e}. Checking OpenCL.")
        except Exception as e: print(f"CUDA check failed: {e}. Checking OpenCL.")
        try: # Check OpenCL
            if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                # Basic check only, full test happens in processor run()
                print("OpenCL detected via cv2.ocl.haveOpenCL().")
                return "OpenCL"
            else: print("OpenCL not available in this OpenCV build.")
        except Exception as e: print(f"OpenCL check failed: {e}.")
        # Fallback to CPU
        print("Falling back to CPU processing.")
        return "CPU"

    def update_processing_mode(self):
        """Updates the mode label asynchronously."""
        self.processing_mode.set("Detecting...")
        def detect_and_update():
            mode = self.get_processing_mode()
            if self.root.winfo_exists():
                self.root.after(0, lambda: self.processing_mode.set(mode))
        threading.Thread(target=detect_and_update, daemon=True).start()

    def on_closing(self, event=None):
        """Handles window close event (X button) and Ctrl+C, ensuring cleanup."""
        print("WM_DELETE_WINDOW / on_closing called.")
        should_exit = True
        if self.is_processing:
            # Ask for confirmation if processing is active
            should_exit = messagebox.askyesno("Exit Confirmation",
                                              "Processing is in progress. Are you sure you want to exit?\nThis will attempt to cancel the current process.")
            if should_exit:
                print("Exit confirmed during processing. Requesting cancellation...")
                # Signal cancellation if not already signalled
                if not self.cancel_event.is_set(): self.cancel_event.set()
                # Disable buttons immediately
                self.start_button.config(state=tk.DISABLED); self.cancel_button.config(state=tk.DISABLED)
                self.set_queue_buttons_state(tk.DISABLED) # Also disable queue buttons
                self.status_text.set("Exiting - Cancelling process...")
                # Give the processing thread a moment to react to the cancel event
                if self.processing_thread and self.processing_thread.is_alive():
                    print("Waiting briefly for processing thread to handle cancel...")
                    self.processing_thread.join(timeout=2.0) # Wait slightly longer
                # Check if thread is still alive after waiting
                if self.processing_thread and self.processing_thread.is_alive():
                     print("Warning: Processing thread still alive after cancel timeout during exit.")
                # Proceed with exit
                print("Destroying root window...")
                self.root.destroy()
            else:
                # User chose not to exit
                print("Exit cancelled by user.")
                return # Abort closing process
        else:
            # No process running, exit cleanly
            print("Exiting application (no process running).")
            self.root.destroy() # Close the window


# --- Main Execution ---
if __name__ == "__main__":
    # Required for multiprocessing support when bundled (e.g., with PyInstaller)
    # Although Pool isn't used, freeze_support() is good practice for bundled apps.
    freeze_support()

    # Initialize Tkinter root window, using themed Tk if available
    root = None
    if THEMED:
        try: root = ThemedTk(theme=DEFAULT_THEME)
        except Exception as e: print(f"Theme Error ({DEFAULT_THEME}): {e}. Falling back."); root = tk.Tk()
    else: root = tk.Tk()

    # Create the main application instance
    app = BackgroundRemoverApp(root)

    # --- Signal Handling (Ctrl+C) ---
    def signal_handler(sig, frame):
        """Handles SIGINT (Ctrl+C) by triggering the on_closing method."""
        print("\nCtrl+C detected. Requesting application close...")
        # Schedule on_closing to run in the main GUI thread
        if root.winfo_exists():
             root.after(0, app.on_closing) # Use after(0) to avoid direct GUI calls from signal handler
        else:
             # If root somehow destroyed, attempt direct call (might not do much)
             try: app.on_closing()
             except: pass

    # Register the signal handler only if it's the default one
    # Avoids issues if run in environments that already handle SIGINT (like some IDEs)
    if signal.getsignal(signal.SIGINT) == signal.default_int_handler:
        try:
            signal.signal(signal.SIGINT, signal_handler)
            print("Registered Ctrl+C handler.")
        except ValueError as e: # Can happen on Windows sometimes
             print(f"Warning: Could not register SIGINT handler: {e}")
    else:
        print("Warning: SIGINT handler already registered, not overriding (Ctrl+C might not trigger clean exit).")

    # --- Start the Tkinter main loop ---
    try:
        print("Starting Tkinter mainloop...")
        root.mainloop()
        print("Tkinter mainloop finished.")
    except KeyboardInterrupt:
        # Catch KeyboardInterrupt if signal handler registration failed or wasn't default
        print("\nKeyboardInterrupt caught in mainloop. Closing.")
        # Ensure cleanup is attempted even with raw KeyboardInterrupt
        if root.winfo_exists():
             app.on_closing()
