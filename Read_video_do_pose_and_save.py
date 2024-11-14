import os
import cv2
import numpy as np
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

def create_detector(model_path):
    """Create a detector instance in each process"""
    protoFile = os.path.join(model_path, "pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile = os.path.join(model_path, "pose_iter_160000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def process_frame(frame, frame_number, frame_size, net, net_input_size=(368, 368), threshold=0.1):
    """Process a single frame"""
    frame_width, frame_height = frame_size
    
    # Create blob
    inpBlob = cv2.dnn.blobFromImage(
        frame,
        1.0 / 255,
        net_input_size,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    
    net.setInput(inpBlob)
    output = net.forward()

    # Calculate scale factors
    scaleX = frame_width / output.shape[3]
    scaleY = frame_height / output.shape[2]

    # Use numpy operations for faster point detection
    points = []
    point_data = []
    
    for i in range(15):  # nPoints = 15
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        x = scaleX * point[0]
        y = scaleY * point[1]
        
        point_data.append([frame_number, i+1, x, y])
        points.append((int(x), int(y)) if prob > threshold else None)

    return points, point_data

def draw_skeleton(frame, points, frame_number, total_frames, fps):
    """Draw the skeleton on the frame"""
    POSE_PAIRS = [
        [0, 1], [1, 2], [2, 3], [3, 4], [1, 5],
        [5, 6], [6, 7], [1, 14], [14, 8], [8, 9],
        [9, 10], [14, 11], [11, 12], [12, 13]
    ]
    
    imSkeleton = np.array(frame, copy=True)
    
    for pair in POSE_PAIRS:
        if points[pair[0]] and points[pair[1]]:
            cv2.line(imSkeleton, points[pair[0]], points[pair[1]], (255, 255, 0), 2)
            cv2.circle(imSkeleton, points[pair[0]], 8, (255, 0, 0), thickness=-1, 
                      lineType=cv2.FILLED)

    cv2.putText(imSkeleton, f'Frame: {frame_number} / {total_frames}', 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 
                cv2.LINE_AA)
    cv2.putText(imSkeleton, f'FPS: {fps}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    return imSkeleton

def process_batch(args):
    """Process a batch of frames in a separate process"""
    frames_data, frame_size, total_frames, fps, model_path = args
    
    # Create detector for this process
    net = create_detector(model_path)
    
    # Enable OpenCV optimizations for this process
    cv2.setNumThreads(4)
    cv2.ocl.setUseOpenCL(True)
    
    results = []
    for frame, frame_num in frames_data:
        points, point_data = process_frame(frame, frame_num, frame_size, net)
        annotated_frame = draw_skeleton(frame, points, frame_num, total_frames, fps)
        results.append((annotated_frame, point_data, frame_num))
    
    return results

def process_video(source_path, output_path, start_frame=50, end_frame=None, num_workers=None, batch_size=8):
    """
    Main video processing function with parallel processing
    """
    mp.freeze_support()  # Handle Windows process spawning
    
    # Use number of CPU cores - 1 for number of workers if not specified
    if num_workers is None:
        num_workers = max(1, min(mp.cpu_count() - 1, 8))  # Limit to 8 workers by default
    
    # Initialize video capture
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError("Error opening video stream or file")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    # Validate and adjust frame range
    start_frame = max(0, min(start_frame, total_frames - 1))
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    else:
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

    frames_to_process = end_frame - start_frame

    print("\nInitializing video processing:")
    print(f"├── Input video: {os.path.basename(source_path)}")
    print(f"├── Output video: {os.path.basename(output_path)}")
    print(f"├── Frame range: {start_frame} to {end_frame} ({frames_to_process} frames)")
    print(f"├── Video properties:")
    print(f"│   ├── Resolution: {frame_width}x{frame_height}")
    print(f"│   ├── FPS: {fps:.2f}")
    print(f"│   └── Total frames: {total_frames}")
    print(f"└── Processing config:")
    print(f"    ├── Workers: {num_workers}")
    print(f"    └── Batch size: {batch_size}")
    print("\nStarting processing...")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_pose = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        frame_size
    )
    
    if not out_pose.isOpened():
        raise ValueError("Failed to initialize video writer. Please check if the codec is available.")

    # Pre-allocate results storage
    point_matrix = []
    processed_frames = {}
    next_frame_to_write = start_frame
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        # Create process pool
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process frames in batches
            frame_batch = []
            batch_futures = []
            
            # Create progress bar for total frames to process
            with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
                start_time = time.time()
                frames_processed = 0
                
                for frame_num in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_batch.append((frame, frame_num))
                    
                    # When batch is full, submit for processing
                    if len(frame_batch) >= batch_size:
                        future = executor.submit(
                            process_batch,
                            (frame_batch, frame_size, frames_to_process, fps, "Model")
                        )
                        batch_futures.append((future, frame_num))
                        frame_batch = []
                    
                    # Process completed batches
                    completed_futures = [(f, n) for f, n in batch_futures if f.done()]
                    for future, _ in completed_futures:
                        results = future.result()
                        batch_futures = [(f, n) for f, n in batch_futures if not f.done()]
                        
                        # Store results
                        for annotated_frame, frame_points, frame_idx in results:
                            processed_frames[frame_idx] = (annotated_frame, frame_points)
                        
                        # Write frames in order
                        while next_frame_to_write in processed_frames:
                            annotated_frame, frame_points = processed_frames.pop(next_frame_to_write)
                            out_pose.write(annotated_frame)
                            point_matrix.extend(frame_points)
                            next_frame_to_write += 1
                            frames_processed += 1
                            
                        # Update progress
                        elapsed_time = time.time() - start_time
                        current_fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
                        pbar.set_postfix({"FPS": f"{current_fps:.2f}"})
                        pbar.update(len(results))
                
                # Process remaining frames
                if frame_batch:
                    future = executor.submit(
                        process_batch,
                        (frame_batch, frame_size, frames_to_process, fps, "Model")
                    )
                    batch_futures.append((future, end_frame))
                
                # Wait for all remaining batches
                for future, _ in batch_futures:
                    results = future.result()
                    for annotated_frame, frame_points, frame_idx in results:
                        processed_frames[frame_idx] = (annotated_frame, frame_points)
                    
                    # Write remaining frames
                    while next_frame_to_write in processed_frames:
                        annotated_frame, frame_points = processed_frames.pop(next_frame_to_write)
                        out_pose.write(annotated_frame)
                        point_matrix.extend(frame_points)
                        next_frame_to_write += 1
                        frames_processed += 1
                
                    # Update final progress
                    elapsed_time = time.time() - start_time
                    final_fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
                    pbar.set_postfix({"FPS": f"{final_fps:.2f}"})
                    pbar.update(len(results))

    finally:
        # Clean up resources
        cap.release()
        out_pose.release()

    # Save point matrix
    output_csv = output_path.rsplit('.', 1)[0] + '_points.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerows(point_matrix)
    
    # Print final statistics
    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    print(f"\nProcessing complete:")
    print(f"Processed {frames_processed} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Points saved to: {output_csv}")

if __name__ == "__main__":
    source = 'Data/20241114_215252.mp4'
    output = 'Data/test.mp4'
    
    # Process frames with parallel processing
    process_video(
        source, 
        output, 
        start_frame=50, 
        end_frame=150, 
        batch_size=8  # Increased batch size for better throughput
    )