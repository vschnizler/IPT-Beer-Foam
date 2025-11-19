import cv2
import threading
import time
import os

# --- Configuration ---
VIDEO_SOURCE_1 = 6  # Corresponds to "/dev/video6"
VIDEO_SOURCE_2 = 2  # Corresponds to "/dev/video4"
OUTPUT_FILENAME_1 = 'output_video_1.avi'
OUTPUT_FILENAME_2 = 'output_video_2.avi'
FRAME_WIDTH = 640   # Set your desired width
FRAME_HEIGHT = 480  # Set your desired height
FPS = 30.0          # Set your desired frame rate
# --- End Configuration ---

def process_video_stream(source_id, window_name, output_filename, width, height, fps):
    """
    Opens a video source, displays it in a named window, and saves it to a file.
    Runs in a separate thread.
    """
    cap = cv2.VideoCapture(source_id)

    if not cap.isOpened():
        print(f"Error: Cannot open camera source {source_id}")
        return

    # Try to set resolution (optional, may not be supported by all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Define the codec and create a VideoWriter object
    # FourCC is a 4-byte code used to specify the video codec. MJPG is a common choice.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_filename}. Check your codec and file path.")
        cap.release()
        return

    print(f"Streaming and saving from source {source_id} to '{output_filename}'. Press 'q' in its window to stop.")
    
    start_time = time.time()
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if not ret:
            print(f"Can't receive frame from source {source_id} (stream end?). Exiting thread...")
            break
        
        # Ensure frame is the expected size before writing (important for saving)
        # Resize if necessary, although it's better to set the correct capture properties
        frame_resized = cv2.resize(frame, (width, height))
        
        # Write the processed frame (color frame in this case)
        out.write(frame_resized)
        
        # Display the resulting frame
        # We display the color frame, not the grayscale, for better visualization
        cv2.imshow(window_name, frame_resized)
        
        # Check for 'q' key press in the specific window
        # The key press will only be registered if the corresponding window is active/focused
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Stop signal 'q' received in window '{window_name}'. Stopping stream...")
            break

    # Release everything when the job is finished
    cap.release()
    out.release()
    # Note: cv2.destroyAllWindows() will be called only once outside the threads
    print(f"Stream from source {source_id} to '{output_filename}' finished.")

def main():
    # Define the arguments for each thread
    # args1 = (VIDEO_SOURCE_1, 'Camera 1 (/dev/video6)', OUTPUT_FILENAME_1, FRAME_WIDTH, FRAME_HEIGHT, FPS)
    args2 = (VIDEO_SOURCE_2, 'Camera 2 (/dev/video1)', OUTPUT_FILENAME_2, FRAME_WIDTH, FRAME_HEIGHT, FPS)


    # Create and start the threads
    # thread1 = threading.Thread(target=process_video_stream, args=args1)
    thread2 = threading.Thread(target=process_video_stream, args=args2)

    # thread1.start()
    thread2.start()

    # Wait for both threads to finish
    # thread1.join()
    thread2.join()

    # Clean up the windows outside the threads
    cv2.destroyAllWindows()
    print("All video streams have been closed and files saved.")

if __name__ == "__main__":
    main()