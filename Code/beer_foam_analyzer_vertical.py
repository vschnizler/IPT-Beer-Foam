import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, Optional
import sys


class FoamAnalyzer:
    """Analyzes beer foam coverage in video frames."""
    
    def __init__(self, video_path: str, output_csv: str = "../Data/foam_data_new.csv", 
                 frame_skip: int = 0, threshold: int = 200, 
                 roi: Optional[Tuple[int, int, int, int]] = None,
                 detect_glass: bool = True,
                 min_glass_radius: int = 50,
                 foam_sensitivity: str = "medium"):
        """
        Initialize the foam analyzer.
        
        Args:
            video_path: Path to the input video file
            output_csv: Path to the output CSV file
            frame_skip: Number of frames to skip between analyses (0 = analyze every frame)
            threshold: Brightness threshold for foam detection (0-255) - overridden by foam_sensitivity
            roi: Optional region of interest (x, y, width, height) to analyze only part of frame
            detect_glass: If True, automatically detect circular glass boundary
            min_glass_radius: Minimum radius for glass detection (pixels)
            foam_sensitivity: Preset sensitivity levels ('low', 'medium', 'high', 'very_high') or 'custom'
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.frame_skip = frame_skip
        self.roi = roi
        self.detect_glass = detect_glass
        self.min_glass_radius = min_glass_radius
        self.glass_mask = None
        
        # Set threshold based on sensitivity level
        sensitivity_presets = {
            'low': 220,        # Only very bright foam
            'medium': 200,     # Default - typical foam
            'high': 180,       # Includes slightly darker foam
            'very_high': 160,  # Catches most lighter areas
            'custom': threshold  # Use custom threshold value
        }
        
        if foam_sensitivity.lower() in sensitivity_presets:
            self.threshold = sensitivity_presets[foam_sensitivity.lower()]
            self.foam_sensitivity = foam_sensitivity.lower()
        else:
            print(f"Warning: Invalid sensitivity '{foam_sensitivity}'. Using 'medium'.")
            self.threshold = sensitivity_presets['medium']
            self.foam_sensitivity = 'medium'
        
        # Validate video file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    def detect_glass_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect the circular boundary of the beer glass.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (center_x, center_y, radius) or None if not detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=self.min_glass_radius,
            maxRadius=min(frame.shape[0], frame.shape[1]) // 2
        )
        
        if circles is not None:
            # Convert to integers
            circles = np.uint16(np.around(circles))
            # Get the first (most prominent) circle
            x, y, r = circles[0][0]
            return (int(x), int(y), int(r))
        
        return None
    
    def create_circular_mask(self, shape: Tuple[int, int], 
                            center: Tuple[int, int], radius: int) -> np.ndarray:
        """
        Create a circular mask.
        
        Args:
            shape: Image shape (height, width)
            center: Center coordinates (x, y)
            radius: Circle radius
            
        Returns:
            Binary mask where 1 = inside circle, 0 = outside
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, -1)
        return mask
    
    def get_foam_area(self, frame: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate the foam coverage area and beer (non-foam) area in a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (foam_area_pixels, beer_area_pixels, total_area_pixels, 
                     foam_percentage, beer_percentage, foam_to_beer_ratio)
        """
        # Apply ROI if specified
        if self.roi:
            x, y, w, h = self.roi
            frame = frame[y:y+h, x:x+w]
        
        # Detect glass circle on first frame or if not yet detected
        if self.detect_glass and self.glass_mask is None:
            circle_data = self.detect_glass_circle(frame)
            if circle_data:
                center_x, center_y, radius = circle_data
                self.glass_mask = self.create_circular_mask(
                    frame.shape, (center_x, center_y), radius
                )
                print(f"Glass detected: center=({center_x}, {center_y}), radius={radius}px")
            else:
                print("Warning: Could not detect glass circle. Analyzing entire frame.")
                self.detect_glass = False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate foam (bright regions)
        _, foam_thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply glass mask if available
        if self.glass_mask is not None:
            # Only count pixels inside the glass
            masked_foam = cv2.bitwise_and(foam_thresh, foam_thresh, mask=self.glass_mask)
            foam_area = np.sum(masked_foam > 0)
            total_area = np.sum(self.glass_mask > 0)
            
            # Beer area is total glass area minus foam area
            beer_area = total_area - foam_area
        else:
            # Count all pixels
            foam_area = np.sum(foam_thresh > 0)
            total_area = foam_thresh.shape[0] * foam_thresh.shape[1]
            beer_area = total_area - foam_area
        
        # Calculate percentages
        foam_percentage = (foam_area / total_area) * 100 if total_area > 0 else 0
        beer_percentage = (beer_area / total_area) * 100 if total_area > 0 else 0
        
        # Calculate foam-to-beer ratio
        foam_to_beer_ratio = (foam_area / beer_area) if beer_area > 0 else float('inf')
        
        return foam_area, beer_area, total_area, foam_percentage, beer_percentage, foam_to_beer_ratio
    
    def process_video(self, show_preview: bool = False) -> pd.DataFrame:
        """
        Process the video and extract foam data.
        
        Args:
            show_preview: If True, display a preview window during processing
            
        Returns:
            DataFrame containing the foam data
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frame skip: {self.frame_skip}")
        print(f"  Analyzing every {self.frame_skip + 1} frame(s)")
        print(f"  Foam sensitivity: {self.foam_sensitivity.upper()}")
        print(f"  Threshold value: {self.threshold}")
        print(f"  Glass detection: {'ON' if self.detect_glass else 'OFF'}")
        if self.roi:
            print(f"  ROI: {self.roi}")
        print()
        
        # Data storage
        data = {
            'frame_number': [],
            'time_seconds': [],
            'foam_area_pixels': [],
            'total_area_pixels': [],
            'foam_percentage': [],
            'beer_area_pixels': [],
            'beer_percentage': [],
            'foam_to_beer_ratio': []
        }
        
        frame_count = 0
        analyzed_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Only analyze frames at specified intervals
                if frame_count % (self.frame_skip + 1) == 0:
                    # Calculate foam area and beer area
                    foam_area, beer_area, total_area, foam_pct, beer_pct, ratio = self.get_foam_area(frame)
                    
                    # Calculate time
                    time_sec = frame_count / fps if fps > 0 else 0
                    
                    # Store data
                    data['frame_number'].append(frame_count)
                    data['time_seconds'].append(time_sec)
                    data['foam_area_pixels'].append(foam_area)
                    data['total_area_pixels'].append(total_area)
                    data['foam_percentage'].append(foam_pct)
                    data['beer_area_pixels'].append(beer_area)
                    data['beer_percentage'].append(beer_pct)
                    data['foam_to_beer_ratio'].append(ratio)
                    
                    analyzed_count += 1
                    
                    # Show preview if requested
                    if show_preview:
                        # Create visualization
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
                            print("\nAnalysis interrupted by user")
                            break
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
                
                frame_count += 1
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames analyzed: {analyzed_count}")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
    
    def save_to_csv(self, df: pd.DataFrame):
        """Save the DataFrame to a CSV file."""
        df.to_csv(self.output_csv, index=False)
        print(f"Data saved to: {self.output_csv}")
        print(f"Total data points: {len(df)}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  FOAM:")
        print(f"    Average foam coverage: {df['foam_percentage'].mean():.2f}%")
        print(f"    Max foam coverage: {df['foam_percentage'].max():.2f}%")
        print(f"    Min foam coverage: {df['foam_percentage'].min():.2f}%")
        print(f"    Initial foam coverage: {df['foam_percentage'].iloc[0]:.2f}%")
        print(f"    Final foam coverage: {df['foam_percentage'].iloc[-1]:.2f}%")
        print(f"  BEER:")
        print(f"    Average beer coverage: {df['beer_percentage'].mean():.2f}%")
        print(f"    Max beer coverage: {df['beer_percentage'].max():.2f}%")
        print(f"    Min beer coverage: {df['beer_percentage'].min():.2f}%")
        print(f"    Initial beer coverage: {df['beer_percentage'].iloc[0]:.2f}%")
        print(f"    Final beer coverage: {df['beer_percentage'].iloc[-1]:.2f}%")
        print(f"  RATIO:")
        # Filter out inf values for average calculation
        valid_ratios = df['foam_to_beer_ratio'][df['foam_to_beer_ratio'] != float('inf')]
        if len(valid_ratios) > 0:
            print(f"    Average foam/beer ratio: {valid_ratios.mean():.3f}")
            print(f"    Max foam/beer ratio: {valid_ratios.max():.3f}")
            print(f"    Min foam/beer ratio: {valid_ratios.min():.3f}")
    
    def run(self, show_preview: bool = False):
        """Run the complete analysis pipeline."""
        print(f"Starting foam analysis on: {self.video_path}\n")
        df = self.process_video(show_preview=show_preview)
        self.save_to_csv(df)
        return df


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract foam coverage data from beer video.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze every frame with automatic glass detection
  python foam_area_analyzer.py beer_video.avi
  
  # Use high sensitivity to detect more foam
  python foam_area_analyzer.py beer_video.avi --foam-sensitivity high
  
  # Use low sensitivity for only bright white foam
  python foam_area_analyzer.py beer_video.avi --foam-sensitivity low
  
  # Skip 4 frames between analyses (analyze every 5th frame)
  python foam_area_analyzer.py beer_video.avi --frame-skip 4
  
  # Custom output file with high sensitivity
  python foam_area_analyzer.py beer_video.avi -o my_data.csv --foam-sensitivity very_high
  
  # Show preview window during processing
  python foam_area_analyzer.py beer_video.avi --preview
  
  # Disable automatic glass detection (analyze full frame)
  python foam_area_analyzer.py beer_video.avi --no-glass-detection
  
  # Use custom threshold (overrides sensitivity presets)
  python foam_area_analyzer.py beer_video.avi --foam-sensitivity custom --threshold 150
  
  # Analyze only a region of interest (x, y, width, height)
  python foam_area_analyzer.py beer_video.avi --roi 100 100 400 400
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default='foam_data.csv',
                       help='Output CSV file path (default: foam_data.csv)')
    parser.add_argument('-f', '--frame-skip', type=int, default=0,
                       help='Number of frames to skip between analyses (default: 0)')
    parser.add_argument('-s', '--foam-sensitivity', type=str, default='medium',
                       choices=['low', 'medium', 'high', 'very_high', 'custom'],
                       help='Foam detection sensitivity: low (220), medium (200), high (180), very_high (160), or custom (default: medium)')
    parser.add_argument('-t', '--threshold', type=int, default=200,
                       help='Custom brightness threshold (0-255). Only used with --foam-sensitivity custom (default: 200)')
    parser.add_argument('-r', '--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                       help='Region of interest as: x y width height')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Show preview window during processing')
    parser.add_argument('--no-glass-detection', action='store_true',
                       help='Disable automatic glass circle detection')
    parser.add_argument('--min-glass-radius', type=int, default=50,
                       help='Minimum radius for glass detection in pixels (default: 50)')
    
    args = parser.parse_args()
    
    try:
        analyzer = FoamAnalyzer(
            video_path=args.video,
            output_csv=args.output,
            frame_skip=args.frame_skip,
            threshold=args.threshold,
            roi=tuple(args.roi) if args.roi else None,
            detect_glass=not args.no_glass_detection,
            min_glass_radius=args.min_glass_radius,
            foam_sensitivity=args.foam_sensitivity
        )
        
        analyzer.run(show_preview=args.preview)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


class FoamAnalyzer:
    """Analyzes beer foam coverage in video frames."""
    
    def __init__(self, video_path: str, output_csv: str = "foam_data.csv", 
                 frame_skip: int = 0, threshold: int = 200, 
                 roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize the foam analyzer.
        
        Args:
            video_path: Path to the input video file
            output_csv: Path to the output CSV file
            frame_skip: Number of frames to skip between analyses (0 = analyze every frame)
            threshold: Brightness threshold for foam detection (0-255)
            roi: Optional region of interest (x, y, width, height) to analyze only part of frame
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.frame_skip = frame_skip
        self.threshold = threshold
        self.roi = roi
        
        # Validate video file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    def get_foam_area(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate the foam coverage area in a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (foam_area_pixels, total_area_pixels, foam_percentage)
        """
        # Apply ROI if specified
        if self.roi:
            x, y, w, h = self.roi
            frame = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate foam (bright regions)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate areas
        foam_area = np.sum(thresh > 0)  # Count white pixels (foam)
        total_area = thresh.shape[0] * thresh.shape[1]  # Total pixels
        foam_percentage = (foam_area / total_area) * 100 if total_area > 0 else 0
        
        return foam_area, total_area, foam_percentage
    
    def process_video(self, show_preview: bool = False) -> pd.DataFrame:
        """
        Process the video and extract foam data.
        
        Args:
            show_preview: If True, display a preview window during processing
            
        Returns:
            DataFrame containing the foam data
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frame skip: {self.frame_skip}")
        print(f"  Analyzing every {self.frame_skip + 1} frame(s)")
        print(f"  Threshold: {self.threshold}")
        if self.roi:
            print(f"  ROI: {self.roi}")
        print()
        
        # Data storage
        data = {
            'frame_number': [],
            'time_seconds': [],
            'foam_area_pixels': [],
            'total_area_pixels': [],
            'foam_percentage': []
        }
        
        frame_count = 0
        analyzed_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Only analyze frames at specified intervals
                if frame_count % (self.frame_skip + 1) == 0:
                    # Calculate foam area
                    foam_area, total_area, foam_pct = self.get_foam_area(frame)
                    
                    # Calculate time
                    time_sec = frame_count / fps if fps > 0 else 0
                    
                    # Store data
                    data['frame_number'].append(frame_count)
                    data['time_seconds'].append(time_sec)
                    data['foam_area_pixels'].append(foam_area)
                    data['total_area_pixels'].append(total_area)
                    data['foam_percentage'].append(foam_pct)
                    
                    analyzed_count += 1
                    
                    # Show preview if requested
                    if show_preview:
                        # Create visualization
                        display_frame = frame.copy()
                        if self.roi:
                            x, y, w, h = self.roi
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Add text overlay
                        text = f"Frame: {frame_count} | Foam: {foam_pct:.1f}%"
                        cv2.putText(display_frame, text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow('Foam Analysis Preview', display_frame)
                        
                        # Press 'q' to quit early
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nAnalysis interrupted by user")
                            break
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
                
                frame_count += 1
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames analyzed: {analyzed_count}")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
    
    def save_to_csv(self, df: pd.DataFrame):
        """Save the DataFrame to a CSV file."""
        df.to_csv(self.output_csv, index=False)
        print(f"Data saved to: {self.output_csv}")
        print(f"Total data points: {len(df)}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Average foam coverage: {df['foam_percentage'].mean():.2f}%")
        print(f"  Max foam coverage: {df['foam_percentage'].max():.2f}%")
        print(f"  Min foam coverage: {df['foam_percentage'].min():.2f}%")
        print(f"  Initial foam coverage: {df['foam_percentage'].iloc[0]:.2f}%")
        print(f"  Final foam coverage: {df['foam_percentage'].iloc[-1]:.2f}%")
    
    def run(self, show_preview: bool = False):
        """Run the complete analysis pipeline."""
        print(f"Starting foam analysis on: {self.video_path}\n")
        df = self.process_video(show_preview=show_preview)
        self.save_to_csv(df)
        return df


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract foam coverage data from beer video.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze every frame
  python foam_area_analyzer.py beer_video.avi
  
  # Skip 4 frames between analyses (analyze every 5th frame)
  python foam_area_analyzer.py beer_video.avi --frame-skip 4
  
  # Custom output file and threshold
  python foam_area_analyzer.py beer_video.avi -o my_data.csv --threshold 180
  
  # Show preview window during processing
  python foam_area_analyzer.py beer_video.avi --preview
  
  # Analyze only a region of interest (x, y, width, height)
  python foam_area_analyzer.py beer_video.avi --roi 100 100 400 400
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default='foam_data.csv',
                       help='Output CSV file path (default: foam_data.csv)')
    parser.add_argument('-f', '--frame-skip', type=int, default=0,
                       help='Number of frames to skip between analyses (default: 0)')
    parser.add_argument('-t', '--threshold', type=int, default=200,
                       help='Brightness threshold for foam detection, 0-255 (default: 200)')
    parser.add_argument('-r', '--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                       help='Region of interest as: x y width height')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Show preview window during processing')
    
    args = parser.parse_args()
    
    try:
        analyzer = FoamAnalyzer(
            video_path=args.video,
            output_csv=args.output,
            frame_skip=args.frame_skip,
            threshold=args.threshold,
            roi=tuple(args.roi) if args.roi else None
        )
        
        analyzer.run(show_preview=args.preview)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())