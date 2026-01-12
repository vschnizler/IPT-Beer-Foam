import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def average_curvature(x, y, max_curvature=None):
    """
    Calculate the average curvature of a closed curve from coordinate points.
    
    Parameters:
    -----------
    x : array-like
        x-coordinates of points on the curve
    y : array-like
        y-coordinates of points on the curve
    max_curvature : float, optional
        Maximum curvature tolerance. Points with curvature exceeding this
        value will be excluded from the average calculation.
    
    Returns:
    --------
    float
        Average absolute curvature of the curve
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Calculate first derivatives using central differences
    # For closed curve, wrap around at boundaries
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    
    dx[1:-1] = (x[2:] - x[:-2]) / 2
    dy[1:-1] = (y[2:] - y[:-2]) / 2
    
    # Handle boundaries with wraparound for closed curve
    dx[0] = (x[1] - x[-1]) / 2
    dy[0] = (y[1] - y[-1]) / 2
    dx[-1] = (x[0] - x[-2]) / 2
    dy[-1] = (y[0] - y[-2]) / 2
    
    # Calculate second derivatives
    ddx = np.zeros_like(x)
    ddy = np.zeros_like(y)
    
    ddx[1:-1] = x[2:] - 2*x[1:-1] + x[:-2]
    ddy[1:-1] = y[2:] - 2*y[1:-1] + y[:-2]
    
    # Handle boundaries with wraparound
    ddx[0] = x[1] - 2*x[0] + x[-1]
    ddy[0] = y[1] - 2*y[0] + y[-1]
    ddx[-1] = x[0] - 2*x[-1] + x[-2]
    ddy[-1] = y[0] - 2*y[-1] + y[-2]
    
    # Calculate curvature at each point
    # κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    curvature = numerator / denominator
    
    # Apply maximum curvature tolerance if specified
    if max_curvature is not None:
        curvature = curvature[curvature <= max_curvature]
    
    # Return average curvature
    return np.mean(curvature)


def calculate_average_curvature_closed(input_file, output_file, max_curvature_tolerance=None):
    """
    Read coordinates from CSV, calculate average curvature per frame, and save results.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with columns: frame_number, x, y
    output_file : str
        Path to output CSV file where results will be saved
    max_curvature : float, optional
        Maximum curvature tolerance. Points with curvature exceeding this
        value will be excluded from the average calculation.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with frame numbers and their corresponding average curvatures
    """
    # Read the CSV file
    df = pd.read_csv(input_file, usecols=['frame_number', 'x', 'y'])
    
    # Get unique frame numbers
    frames = df['frame_number'].unique()
    
    # Initialize list to store results
    results = []
    
    # Iterate through each frame
    for frame in frames:
        # Get points for this frame
        frame_data = df[df['frame_number'] == frame]
        x = frame_data['x'].values
        y = frame_data['y'].values
        
        # Calculate average curvature
        if len(x) >= 3:  # Need at least 3 points to calculate curvature
            avg_curv = average_curvature(x, y, max_curvature=max_curvature_tolerance)
            results.append({'frame': frame, 'avg_curvature': avg_curv})
       #ä else:
            # print(f"Warning: Frame {frame} has fewer than 3 points, skipping...")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Write to output file
    results_df.to_csv(output_file, index=False)
    
    # print(f"Processed {len(results)} frames")
    # print(f"Results saved to {output_file}")
    
    return results_df

# def calculate_menger_curvature(p1, p2, p3):
#     """Calculates curvature for a triplet of points."""
#     a = np.linalg.norm(p1 - p2)
#     b = np.linalg.norm(p2 - p3)
#     c = np.linalg.norm(p3 - p1)
    
#     # Area using cross product for 2D points
#     area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    
#     if a * b * c == 0:
#         return 0.0
#     return (4 * area) / (a * b * c)

# def calculate_average_curvature_closed(input_file, output_file, max_curvature_tolerance):
#     df = pd.read_csv(input_file, usecols=['frame_number', 'x', 'y'])
#     results = []

#     for frame, group in df.groupby('frame_number'):
#         # Convert group to list of points (x, y)
#         points = group[['x', 'y']].values.tolist()
        
#         if len(points) < 3:
#             results.append({'frame': frame, 'avg_curvature': 0.0})
#             continue

#         valid_curvatures = []
#         i = 0
        
#         # We use a while loop because the list length changes if we discard points
#         while i < len(points) and len(points) >= 3:
#             # Indices for closed loop: (i-1, i, i+1)
#             # This ensures every point is treated as a 'middle' vertex
#             p_prev = np.array(points[i - 1])
#             p_curr = np.array(points[i])
#             p_next = np.array(points[(i + 1) % len(points)])
            
#             kappa = calculate_menger_curvature(p_prev, p_curr, p_next)
            
#             if kappa <= max_curvature_tolerance:
#                 valid_curvatures.append(kappa)
#                 i += 1
#             else:
#                 # Discard the current point and do not increment i
#                 # This allows the 'new' triplet formed at this index to be checked
#                 del points[i]
#                 # If we delete the last point, we need to break to avoid index error
#                 if i >= len(points):
#                     break
                    
#         avg_k = np.mean(valid_curvatures) if valid_curvatures else 0.0
#         results.append({'frame': frame, 'avg_curvature': avg_k})

#     output_df = pd.DataFrame(results)
#     output_df.to_csv(output_file, index=False)
#     # print(f"Closed-loop processing complete. Saved to {output_file}")

def visualize_curvature_filter(input_file, frame_to_plot, max_curvature_tolerance):
    """
    Plots the original points vs the points that survive the curvature filter
    for a single specific frame.
    """
    df = pd.read_csv(input_file)
    frame_data = df[df['frame_number'] == frame_to_plot]
    
    if frame_data.empty:
        # print(f"No data found for frame {frame_to_plot}")
        return

    # Extract original points
    original_points = frame_data[['x', 'y']].values.tolist()
    points = [np.array(p) for p in original_points]
    
    # Run the filtering logic
    filtered_points = list(points)
    i = 0
    while i < len(filtered_points) and len(filtered_points) >= 3:
        p_prev = filtered_points[i - 1]
        p_curr = filtered_points[i]
        p_next = filtered_points[(i + 1) % len(filtered_points)]
        
        kappa = calculate_menger_curvature(p_prev, p_curr, p_next)
        
        if kappa <= max_curvature_tolerance:
            i += 1
        else:
            del filtered_points[i]
    
    # Convert to arrays for plotting
    orig_arr = np.array(original_points)
    filt_arr = np.array(filtered_points)

    # Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Plot original points (as a light grey line/dots)
    plt.plot(orig_arr[:, 0], orig_arr[:, 1], 'ro-', alpha=1, label='Original (Noisy)')
    
    # Plot filtered points (as a solid blue line)
    # Re-append the first point to the end to close the loop visually
    if len(filt_arr) > 0:
        filt_loop = np.vstack([filt_arr, filt_arr[0]])
        plt.plot(filt_loop[:, 0], filt_loop[:, 1], 'b-o', linewidth=2, label='Filtered (Clean)')

    plt.title(f"Edge Curvature Filter (Frame {frame_to_plot}, Tolerance {max_curvature_tolerance})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.axis('equal') # Keep aspect ratio square
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
