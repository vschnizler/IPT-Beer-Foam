"""
Beer Foam Analyzer Runner
A simplified interface to run foam analysis on beer videos with configurable parameters.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import cv2
import numpy as np

# Try to import - if there are multiple classes, we'll define our own
try:
    from beer_foam_analyzer_vertical import FoamAnalyzer
    # Check if it has the enhanced features
    import inspect
    sig = inspect.signature(FoamAnalyzer.__init__)
    if 'detect_glass' not in sig.parameters or 'foam_sensitivity' not in sig.parameters:
        raise ImportError("FoamAnalyzer doesn't have required features")
except (ImportError, AttributeError):
    # Define the enhanced FoamAnalyzer class here
    # print("Warning: Using embedded FoamAnalyzer class (enhanced version)")
    
    class FoamAnalyzer:
        """Analyzes beer foam coverage in video frames."""
        
        def __init__(self, video_path: str, output_csv: str = "foam_data.csv", 
                     frame_skip: int = 0, threshold: int = 200, 
                     roi: Optional[Tuple[int, int, int, int]] = None,
                     detect_glass: bool = True,
                     min_glass_radius: int = 50,
                     foam_sensitivity: str = "medium",
                     low_bnd: int = 1,
                     high_bnd:int = 15000):
            
            self.video_path = video_path
            self.output_csv = output_csv
            self.frame_skip = frame_skip
            self.roi = roi
            self.detect_glass = detect_glass
            self.min_glass_radius = min_glass_radius
            self.glass_mask = None
            self.low_bnd = low_bnd
            self.high_bnd = high_bnd
            
            # Set threshold based on sensitivity level
            sensitivity_presets = {
                'low': 220,
                'medium': 200,
                'high': 180,
                'very_high': 160,
                'custom': threshold
            }
            
            if foam_sensitivity.lower() in sensitivity_presets:
                self.threshold = sensitivity_presets[foam_sensitivity.lower()]
                self.foam_sensitivity = foam_sensitivity.lower()
            else:
                # print(f"Warning: Invalid sensitivity '{foam_sensitivity}'. Using 'medium'.")
                self.threshold = sensitivity_presets['medium']
                self.foam_sensitivity = 'medium'
            
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
        
        def detect_glass_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                param1=50, param2=30,
                minRadius=self.min_glass_radius,
                maxRadius=min(frame.shape[0], frame.shape[1]) // 2
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                x, y, r = circles[0][0]
                return (int(x), int(y), int(r))
            return None
        
        def create_circular_mask(self, shape: Tuple[int, int], 
                                center: Tuple[int, int], radius: int) -> np.ndarray:
            mask = np.zeros(shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 1, -1)
            return mask
        
        def get_foam_area(self, frame: np.ndarray) -> Tuple[float, float, float, float, float, float]:
            if self.roi:
                x, y, w, h = self.roi
                frame = frame[y:y+h, x:x+w]
            
            if self.detect_glass and self.glass_mask is None:
                circle_data = self.detect_glass_circle(frame)
                if circle_data:
                    center_x, center_y, radius = circle_data
                    self.glass_mask = self.create_circular_mask(
                        frame.shape, (center_x, center_y), radius
                    )
                    # print(f"Glass detected: center=({center_x}, {center_y}), radius={radius}px")
                else:
                    # print("Warning: Could not detect glass circle. Analyzing entire frame.")
                    self.detect_glass = False
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, foam_thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
            
            if self.glass_mask is not None:
                masked_foam = cv2.bitwise_and(foam_thresh, foam_thresh, mask=self.glass_mask)
                foam_area = np.sum(masked_foam > 0)
                total_area = np.sum(self.glass_mask > 0)
                beer_area = total_area - foam_area
            else:
                foam_area = np.sum(foam_thresh > 0)
                total_area = foam_thresh.shape[0] * foam_thresh.shape[1]
                beer_area = total_area - foam_area
            
            foam_percentage = (foam_area / total_area) * 100 if total_area > 0 else 0
            beer_percentage = (beer_area / total_area) * 100 if total_area > 0 else 0
            foam_to_beer_ratio = (foam_area / beer_area) if beer_area > 0 else float('inf')
            
            return foam_area, beer_area, total_area, foam_percentage, beer_percentage, foam_to_beer_ratio
        
        def process_video(self, show_preview: bool = False) -> pd.DataFrame:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {self.video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # print(f"Video properties:")
            # print(f"  FPS: {fps}")
            # print(f"  Total frames: {total_frames}")
            # print(f"  Frame skip: {self.frame_skip}")
            # print(f"  Foam sensitivity: {self.foam_sensitivity.upper()}")
            # print(f"  Threshold value: {self.threshold}")
            # print(f"  Glass detection: {'ON' if self.detect_glass else 'OFF'}")
            
            data = {
                'frame_number': [], 'time_seconds': [],
                'foam_area_pixels': [], 'total_area_pixels': [],
                'foam_percentage': [], 'beer_area_pixels': [],
                'beer_percentage': [], 'foam_to_beer_ratio': []
            }
            
            frame_count = 0
            analyzed_count = 0
            
            try:
                if self.low_bnd > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.low_bnd)
                    # print(f"Skipping to frame {self.low_bnd}...")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    # 2. Safety Break: Stop if we exceed the high boundary
                    if self.high_bnd is not None and current_frame > self.high_bnd:
                        # print(f"Reached high boundary ({self.high_bnd}). Stopping.")
                        break
                    
                    if frame_count % (self.frame_skip + 1) == 0:
                        foam_area, beer_area, total_area, foam_pct, beer_pct, ratio = self.get_foam_area(frame)
                        time_sec = frame_count / fps if fps > 0 else 0
                        
                        data['frame_number'].append(frame_count)
                        data['time_seconds'].append(time_sec)
                        data['foam_area_pixels'].append(foam_area)
                        data['total_area_pixels'].append(total_area)
                        data['foam_percentage'].append(foam_pct)
                        data['beer_area_pixels'].append(beer_area)
                        data['beer_percentage'].append(beer_pct)
                        data['foam_to_beer_ratio'].append(ratio)
                        
                        analyzed_count += 1
                        
                        if show_preview:
                            display_frame = frame.copy()
                            
                            # Get the foam mask for visualization
                            roi_frame = frame
                            if self.roi:
                                x, y, w, h = self.roi
                                roi_frame = frame[y:y+h, x:x+w]
                            
                            # Convert to grayscale and threshold to get foam
                            gray_vis = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                            _, thresh_vis = cv2.threshold(gray_vis, self.threshold, 255, cv2.THRESH_BINARY)
                            
                            # Apply glass mask if available
                            if self.glass_mask is not None:
                                thresh_vis = cv2.bitwise_and(thresh_vis, thresh_vis, mask=self.glass_mask)
                            
                            # Find contours of foam regions
                            foam_contours, _ = cv2.findContours(
                                thresh_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            
                            # Draw foam contours in RED
                            if foam_contours:
                                cv2.drawContours(display_frame, foam_contours, -1, (0, 0, 255), 2)
                            
                            # Draw glass circle if detected (in GREEN)
                            if self.glass_mask is not None:
                                # Find the circle parameters from the mask
                                glass_contours, _ = cv2.findContours(
                                    self.glass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                )
                                if glass_contours:
                                    (x, y), radius = cv2.minEnclosingCircle(glass_contours[0])
                                    cv2.circle(display_frame, (int(x), int(y)), 
                                             int(radius), (0, 255, 0), 2)
                            
                            # Draw ROI if specified (in BLUE)
                            if self.roi:
                                x, y, w, h = self.roi
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            
                            # Add text overlay with both foam and beer percentages
                            text1 = f"Frame: {frame_count} | Foam: {foam_pct:.1f}% | Beer: {beer_pct:.1f}%"
                            text2 = f"Ratio (F/B): {ratio:.3f}" if ratio != float('inf') else "Ratio (F/B): inf"
                            cv2.putText(display_frame, text1, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(display_frame, text2, (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            cv2.imshow('Foam Analysis Preview', display_frame)
                            
                            # Press 'q' to quit early
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                # print("\nAnalysis interrupted by user")
                                break
                    
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        # print(f"Progress: {progress:.1f}%", end='\r')
                    
                    frame_count += 1
            
            finally:
                cap.release()
                if show_preview:
                    cv2.destroyAllWindows()
            
            # print(f"\nProcessing complete! Analyzed {analyzed_count} frames")
            return pd.DataFrame(data)
        
        def save_to_csv(self, df: pd.DataFrame):
            df.to_csv(self.output_csv, index=False)
            # print(f"Data saved to: {self.output_csv}")
            # print(f"\nSummary Statistics:")
            # print(f"  Average foam: {df['foam_percentage'].mean():.2f}%")
            # print(f"  Max foam: {df['foam_percentage'].max():.2f}%")
            # print(f"  Min foam: {df['foam_percentage'].min():.2f}%")
        
        def run(self, show_preview: bool = False):
            # print(f"Starting foam analysis on: {self.video_path}\n")
            df = self.process_video(show_preview=show_preview)
            self.save_to_csv(df)
            return df


def analyze_beer_foam(
    video_path: str,
    output_csv: Optional[str] = None,
    sensitivity: str = "custom",
    frame_skip: int = 0,
    show_preview: bool = False,
    detect_glass: bool = True,
    min_glass_radius: int = 50,
    roi: Optional[Tuple[int, int, int, int]] = None,
    custom_threshold: Optional[int] = 100,
    low_bnd: int = 1,
    high_bnd: int = 20000
) -> pd.DataFrame:
    """
    Analyze beer foam coverage in a video file.
    
    Args:
        video_path: Path to the input video file (e.g., "beer_video.avi")
        output_csv: Path for output CSV file. If None, auto-generates name based on video file
        sensitivity: Foam detection sensitivity - options:
                    - 'low': Only very bright foam (threshold 220)
                    - 'medium': Typical foam detection (threshold 200) [DEFAULT]
                    - 'high': Includes slightly darker foam (threshold 180)
                    - 'very_high': Catches most lighter areas (threshold 160)
                    - 'custom': Use custom_threshold value
        frame_skip: Number of frames to skip between analyses (0 = analyze every frame)
        show_preview: If True, displays a preview window during processing
        detect_glass: If True, automatically detects circular glass boundary
        min_glass_radius: Minimum radius for glass detection in pixels
        roi: Optional region of interest as (x, y, width, height) tuple
        custom_threshold: Custom brightness threshold (0-255), only used when sensitivity='custom'
    
    Returns:
        pandas DataFrame containing the foam analysis data
    
    Examples:
        # Basic analysis with default settings
        df = analyze_beer_foam("my_beer.avi")
        
        # High sensitivity analysis with preview
        df = analyze_beer_foam("my_beer.avi", sensitivity="high", show_preview=True)
        
        # Analyze every 5th frame with custom output file
        df = analyze_beer_foam("my_beer.avi", frame_skip=4, output_csv="results.csv")
        
        # Use custom threshold and specific region of interest
        df = analyze_beer_foam("my_beer.avi", sensitivity="custom", 
                              custom_threshold=150, roi=(100, 100, 400, 400))
        
        # Without automatic glass detection
        df = analyze_beer_foam("my_beer.avi", detect_glass=False)
    """
    
    # Validate video file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Auto-generate output CSV name if not provided
    if output_csv is None:
        video_stem = Path(video_path).stem
        output_csv = f"{video_stem}_foam_data.csv"
    
    # Set threshold for custom sensitivity
    threshold = custom_threshold if sensitivity == 'custom' and custom_threshold is not None else 200
    
    # Create analyzer instance
    analyzer = FoamAnalyzer(
        video_path=video_path,
        output_csv=output_csv,
        frame_skip=frame_skip,
        threshold=threshold,
        roi=roi,
        detect_glass=detect_glass,
        min_glass_radius=min_glass_radius,
        foam_sensitivity=sensitivity,
        low_bnd=low_bnd,
        high_bnd=high_bnd
    )
    
    # Run analysis
    # print(f"=" * 60)
    # print(f"BEER FOAM ANALYSIS")
    # print(f"=" * 60)
    df = analyzer.run(show_preview=show_preview)
    # print(f"=" * 60)
    
    return df


def quick_analysis(video_path: str, sensitivity: str = "medium") -> pd.DataFrame:
    """
    Quick analysis with minimal configuration.
    
    Args:
        video_path: Path to the video file
        sensitivity: 'low', 'medium', 'high', or 'very_high'
    
    Returns:
        DataFrame with foam analysis results
    """
    return analyze_beer_foam(video_path, sensitivity=sensitivity)


def analyze_edge(
    video_path: str,
    target: str,
    sensitivity: str = "medium",
    step_sens: float = 1.0,
    detect_glass: bool = True,
    min_glass_radius: int = 50,
    frame_skip: int = 0,
    preview: bool = False,
    custom_threshold: Optional[int] = None,
    edge_type: str = "inner",
    survival_threshold: int = 1,
    tracking_distance: float = 50.0,
    size_threshold: float = 100.0,
    low_bnd: int = 0,
    high_bnd: Optional[int] = None
) -> dict:
    """
    Analyze the edge of beer foam and extract coordinates within a specific frame interval.
    """
    # Validate inputs and setup paths
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Determine brightness threshold
    sensitivity_presets = {
        'low': 220, 'medium': 200, 'high': 180, 'very_high': 160,
        'custom': custom_threshold if custom_threshold is not None else 200
    }
    threshold = sensitivity_presets.get(sensitivity.lower(), 200)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. Fast-forward to the start interval
    if low_bnd > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, low_bnd)
        # print(f"Skipping to frame {low_bnd}...")

    # print(f"=" * 60)
    # print(f"FOAM EDGE ANALYSIS")
    # print(f"Interval: {low_bnd} to {high_bnd if high_bnd else total_frames}")
    # print(f"=" * 60)
    
    contour_data = {}  
    tracked_contours = []  
    next_track_id = 0
    glass_mask = None
    glass_center = None
    glass_radius = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use actual video frame index for accurate boundary checking
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Check if we have exceeded the high boundary
            if high_bnd is not None and current_frame > high_bnd:
                # print(f"\nReached high boundary ({high_bnd}). Stopping.")
                break

            # Only analyze frames at specified intervals relative to the start
            if (current_frame - low_bnd) % (frame_skip + 1) == 0:
                time_sec = current_frame / fps if fps > 0 else 0
                
                # Glass detection on the first analyzed frame
                if detect_glass and glass_mask is None:
                    gray_glass = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray_glass, (9, 9), 2)
                    circles = cv2.HoughCircles(
                        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                        param1=50, param2=30, minRadius=min_glass_radius,
                        maxRadius=min(frame.shape[0], frame.shape[1]) // 2
                    )
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        glass_center = (int(circles[0][0][0]), int(circles[0][0][1]))
                        glass_radius = int(circles[0][0][2])
                        glass_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.circle(glass_mask, glass_center, glass_radius, 1, -1)
                        # print(f"Glass detected at frame {current_frame}: center={glass_center}, radius={glass_radius}px")

                # Thresholding and contour finding
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, foam_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                if glass_mask is not None:
                    foam_thresh = cv2.bitwise_and(foam_thresh, foam_thresh, mask=glass_mask)
                
                contours, hierarchy = cv2.findContours(foam_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                current_detections = []
                if hierarchy is not None and len(contours) > 0:
                    hierarchy = hierarchy[0]
                    for i, contour in enumerate(contours):
                        if cv2.contourArea(contour) < size_threshold:
                            continue
                        
                        parent_idx = hierarchy[i][3]
                        is_inner = parent_idx != -1
                        is_outer = parent_idx == -1
                        
                        if (edge_type == "inner" and not is_inner) or (edge_type == "outer" and not is_outer):
                            continue
                        
                        M = cv2.moments(contour)
                        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
                        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
                        current_detections.append({'contour': contour, 'center': (cx, cy), 'is_inner': is_inner})

                # Tracking Logic
                matched_tracks = set()
                for det in current_detections:
                    det_center = det['center']
                    best_match_idx = None
                    min_dist = tracking_distance
                    
                    for i, track in enumerate(tracked_contours):
                        if i in matched_tracks: continue
                        dist = np.sqrt((det_center[0]-track['center'][0])**2 + (det_center[1]-track['center'][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_match_idx = i
                    
                    if best_match_idx is not None:
                        track_id = tracked_contours[best_match_idx]['id']
                        matched_tracks.add(best_match_idx)
                        tracked_contours[best_match_idx].update({'center': det_center, 'last_seen': current_frame})
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                        tracked_contours.append({'id': track_id, 'center': det_center, 'last_seen': current_frame})
                    
                    # Store Points
                    if track_id not in contour_data:
                        contour_data[track_id] = {k: [] for k in ['frame_number', 'time_seconds', 'point_index', 'x', 'y', 'distance_from_center', 'edge_type']}
                    
                    epsilon = step_sens * cv2.arcLength(det['contour'], True) / 100
                    approx = cv2.approxPolyDP(det['contour'], epsilon, True)
                    for idx, pt in enumerate(approx):
                        px, py = pt[0]
                        dist_center = np.sqrt((px-glass_center[0])**2 + (py-glass_center[1])**2) if glass_center else 0
                        if glass_radius and abs(dist_center - glass_radius) < 5: continue
                        
                        contour_data[track_id]['frame_number'].append(current_frame)
                        contour_data[track_id]['time_seconds'].append(time_sec)
                        contour_data[track_id]['point_index'].append(idx)
                        contour_data[track_id]['x'].append(int(px))
                        contour_data[track_id]['y'].append(int(py))
                        contour_data[track_id]['distance_from_center'].append(float(dist_center))
                        contour_data[track_id]['edge_type'].append('inner' if det['is_inner'] else 'outer')

                tracked_contours = [t for t in tracked_contours if current_frame - t['last_seen'] < 10]

                if preview:
                    cv2.imshow('Foam Edge Analysis Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            
          #  if current_frame % 100 == 0:
                # print(f"Progress: {(current_frame/total_frames)*100:.1f}% | Frame: {current_frame}", end='\r')

    finally:
        cap.release()
        if preview: cv2.destroyAllWindows()
    
    return _save_contour_data(contour_data, target_path, survival_threshold)


def _save_contour_data(contour_data: dict, target_path: Path, survival_threshold: int = 1) -> dict:
    """
    Helper function to save contour data to separate CSV files.
    Only saves contours that appear in at least survival_threshold frames.
    
    Args:
        contour_data: Dictionary of contour data
        target_path: Path to output folder
        survival_threshold: Minimum number of frames a contour must appear in to be saved
        
    Returns:
        Dictionary mapping contour indices to their DataFrames
    """
    result_dfs = {}
    filtered_count = 0
    saved_count = 0
    
    # print(f"\nSaving contours to CSV files (survival threshold: {survival_threshold} frames)...")
    
    for contour_idx, data in contour_data.items():
        # Skip empty contours
        if len(data['frame_number']) == 0:
            continue
        
        # Create DataFrame for this contour
        df = pd.DataFrame(data)
        
        # Count unique frames this contour appears in
        unique_frames = df['frame_number'].nunique()
        
        # Check if contour meets survival threshold
        if unique_frames < survival_threshold:
            filtered_count += 1
            continue
        
        # Save to CSV with index as filename
        csv_filename = target_path / f"{contour_idx}.csv"
        df.to_csv(csv_filename, index=False)
        
        result_dfs[contour_idx] = df
        saved_count += 1
        
        # print(f"  Saved contour {contour_idx}: {len(df)} points across {unique_frames} frames -> {csv_filename.name}")
    
    # print(f"\nSummary:")
    # print(f"  Total contours analyzed: {len(contour_data)}")
    # print(f"  Contours saved: {saved_count}")
    # print(f"  Contours filtered out: {filtered_count} (< {survival_threshold} frames)")
    # print(f"  Output folder: {target_path.absolute()}")
    
    if len(result_dfs) > 0:
        total_points = sum(len(df) for df in result_dfs.values())
        avg_frames = sum(df['frame_number'].nunique() for df in result_dfs.values()) / len(result_dfs)
        # print(f"  Total edge points: {total_points}")
        # print(f"  Average points per contour: {total_points / len(result_dfs):.1f}")
        # print(f"  Average frames per contour: {avg_frames:.1f}")
    
    return result_dfs


def batch_analyze_videos(
    video_paths: list,
    sensitivity: str = "medium",
    frame_skip: int = 0,
    output_dir: Optional[str] = None
) -> dict:
    """
    Analyze multiple beer videos in batch.
    
    Args:
        video_paths: List of video file paths
        sensitivity: Foam detection sensitivity for all videos
        frame_skip: Frames to skip for all videos
        output_dir: Directory to save all CSV files. If None, saves in current directory
    
    Returns:
        Dictionary mapping video paths to their DataFrames
    """
    results = {}
    
    for i, video_path in enumerate(video_paths, 1):
        # print(f"\n\nProcessing video {i}/{len(video_paths)}: {video_path}")
        # print("-" * 60)
        
        try:
            # Generate output path
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                video_name = Path(video_path).stem
                output_csv = str(Path(output_dir) / f"{video_name}_foam_data.csv")
            else:
                output_csv = None
            
            # Analyze video
            df = analyze_beer_foam(
                video_path=video_path,
                output_csv=output_csv,
                sensitivity=sensitivity,
                frame_skip=frame_skip
            )
            
            results[video_path] = df
            
        except Exception as e:
            # print(f"Error processing {video_path}: {e}")
            results[video_path] = None
    
    return results

