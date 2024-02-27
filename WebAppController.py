import threading
import multiprocessing
import time
import datetime
import serial
from azure.iot.device import IoTHubDeviceClient, MethodResponse


import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import math
import numpy as np
import cv2
import serial
import time
import math
import torch

# Required install
# pip install azure-iot-hub

# Additonal info : resp_status : 200 = success
#                              : 500 = unsuccessful
#                              : 404 = method not found                               

# Do not modify
# Connection string responsible for securing connection for azure iot c2d connection
CONNECTION_STRING = "HostName=GaiaBot-rasberryPi.azure-devices.net;DeviceId=rasberryPi2024;SharedAccessKey=iBvdRg7UgMhgOb6fR/NUF1yYGXo1MVz8rAIoTMi0WKg="
x = True
on = False
off = False
executed = False
returnBot = False
 
# Client responsible for initialsing a opening on the device
def create_client():
    global executed
    global x
    global on
    global off
    global returnBot
    
    # Instantiate the client
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)

    # Define the handler for method requests
    def method_request_handler(method_request):
        global executed
        global x
        global on
        global off
        global returnBot
        if method_request.name == "StartDevice":
            # Act on the method by starting the device
            print("Starting Device")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            
            # Make sure property names have no spaces in them
            reported_props = {"startTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # For now it sends back the current time to the cloud
            
            #client.patch_twin_reported_properties(reported_props)
            #print( "Device twins updated with latest start time")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull return 500
            
            off = False;
            on = True;
            executed = False;
            print(on)
            
            
            resp_status = 200
            
            # Payload has to be a json property
            resp_payload = {"Response": "This is the response from the device",
                            "startTime": current_time} 
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
            
        elif method_request.name == "StopDevice":
             # Act on the method by stopping the device
            print("Stopping Device")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            # reported_props = {"stopTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # client.patch_twin_reported_properties(reported_props)
            # print("Device twins updated with latest rebootTime")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull set resp_status to 500
            
            off = True;
            on = False;
            executed = False;
            
            resp_status = 200
            
            # Look above for on method to see info about payload
            resp_payload = {"Response": "This is the response from the device",
                            "Time": current_time}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
            client.shutdown()
            
            
        # Add elif to continue expanding number of methods / for multiple methods switch to switch rather than if's
            
        else:
            # Create a method response indicating the method request was for an unknown method
            resp_status = 404
            resp_payload = {"Response": "Unknown method"}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)

        # Send the method response
        client.send_method_response(method_response)

    try:
        # Attach the handler to the client
        client.on_method_request_received = method_request_handler
    except:
        # In the event of failure, clean up
        client.shutdown()

    return client

def main():
    global executed
    global x
    global on
    global off
    global returnBot
    
    print ("Starting the Iot hub for the device")
    client = create_client()

    print ("Waiting for commands, press Ctrl-C to exit")
    try:
        #This should be set to false when the device stops/loses connection
        # Wait for program exit
        
        main_loop_thread = threading.Thread(target=mainWhileLoop)
        main_loop_thread.start()

        # Wait for the main loop thread to finish (this won't happen in this example since the loop runs indefinitely)
        main_loop_thread.join()
        
        
    except KeyboardInterrupt:
        print("IoTHubDeviceClient sample stopped")
    finally:
        # Graceful exit
        print("Shutting down IoT Hub Client")
        client.shutdown()
        
def object_detection():

    webcontroller()

def mainWhileLoop():
    global executed
    global x
    global on
    global off
    global returnBot
    object_detection_thread = multiprocessing.Process(target=object_detection, args=())
    
    while x:
            if(on):
                if(not executed):
                    print(True)
                    executed = True
                    object_detection_thread.start()                
                    
            elif(off):        
                if(not executed):
                    executed = True
                    x = False
                    object_detection_thread.terminate()                
                    #Turn the whole system off
            elif(returnBot):
                if(not executed):
                    executed = True
                    #Return the bot back to base
            
            #Rate of refresh - cant be to short nor to long
            time.sleep(1)
      

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Read camera calibration results
calibration_data = np.load('yolov5\calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeff = calibration_data['dist_coeff']
rvecs = calibration_data['rvecs']
tvecs = calibration_data['tvecs']

def img_to_world(img_point):
    # Image coordinates
    img_point = np.array([img_point], dtype=np.float32)

    # Undistort image points
    undistorted_point = cv2.undistortPoints(img_point, camera_matrix, dist_coeff)

    # Convert camera coordinates to 3D coordinates
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])  # Assume only one calibration image is used
    translation_vector = tvecs[0]  # Assume only one calibration image is used
    camera_point = undistorted_point[0][0]  # Extracting camera point from the array
    world_point = np.dot(np.linalg.inv(rotation_matrix), (camera_point - translation_vector))

    return world_point

def estimate_distance(object_height_pixels, real_height, focal_length):
    # Use the principle of similar triangles to estimate the distance from the object to the camera
    distance = (real_height * focal_length) / object_height_pixels
    return distance

# Focal length obtained from calibration (in millimeters)
focal_length_mm = camera_matrix[0,0]  # Example: 4 millimeters

# Real height of the object (in the same units as the focal length)
real_height_mm = 60  # Example: 20 millimeters

def pixel_to_real_center(pixel_center, image_size, distance_to_object, focal_length):
    # Image dimensions (width, height)
    image_width, image_height = image_size

    # Pixel coordinates of the image center
    pixel_center_x, pixel_center_y = pixel_center

    # Calculate horizontal and vertical field of view (FOV) in radians
    fov_x = 2 * np.arctan(image_width / (2 * focal_length))
    fov_y = 2 * np.arctan(image_height / (2 * focal_length))

    # Calculate the real-world coordinates of the image center
    real_center_x = (pixel_center_x - image_width / 2) * (distance_to_object / focal_length) * np.tan(fov_x / 2)
    real_center_y = (pixel_center_y - image_height / 2) * (distance_to_object / focal_length) * np.tan(fov_y / 2)

    return real_center_x, real_center_y

def undistorted(x,y):
    pixel_point = np.array([[x, y]], dtype=np.float32)
    pixel_point_homogeneous = np.hstack((pixel_point, np.ones((pixel_point.shape[0], 1)))).astype(np.float32)
    undistorted_point_homogeneous = cv2.undistortPoints(pixel_point_homogeneous, camera_matrix, dist_coeff)
    x_un, y_un = undistorted_point_homogeneous[:, :2].reshape(-1)
    return x_un, y_un


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file

        save_csv = True
        csv_file_path = r"C:\Users\erenz\Downloads\yolov5_2\detections.csv"

# Ensure directory exists
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        # Check if the file exists and has content to decide on writing headers
        file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0

        with open(csv_file_path, 'a' if file_exists else 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                # Write the header row only if the file does not exist or is empty
                csv_writer.writerow(["frame", "class", "x_center", "y_center", "z_distance", "confidence"])
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if confidence > 0.60:  # Only proceed if confidence is greater than 0.60
                            x1, y1, x2, y2 = map(int, xyxy)  # Convert bounding box coordinates to integers
                            h = np.abs(y2 - y1)  # Calculate the height of the bounding box
                            
                            # Estimate the distance based on the bounding box height, real object height, and camera focal length
                            dis = estimate_distance(h, real_height=real_height_mm, focal_length=focal_length_mm) - 4
                            
                            # Calculate real-world X, Y center positions based on pixel positions and estimated distance
                            x, y = pixel_to_real_center(((x1 + x2) / 2, (y1 + y2) / 2), (640, 640), dis, focal_length=focal_length_mm)
                            print(f"Coordinates: x={x}, y={y}, z={dis}")  # Print the calculated real-world coordinates

                            roboticArm(y,dis)
                            # Write the detection information to the CSV file if save_csv flag is True
                            if save_csv:
                                # Prepare the row data
                                row_data = [frame, names[c], x, y, dis, f"{confidence:.2f}"]
                                
                                # Write the row data to the CSV
                                csv_writer.writerow(row_data)

                            
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # xy = xywh[:2]
                            x1, y1, x2, y2 = xyxy
                            # x1_un, y1_un = undistorted(x1,y1)
                            # x2_un, y2_un = undistorted(x2,y2)
                            h = np.abs(y2 - y1)
                            dis = estimate_distance(h,real_height=real_height_mm,focal_length=focal_length_mm) - 4
                            x, y = pixel_to_real_center(((x1+x2)/2, (y1+y2)/2), (640, 640), dis, focal_length=focal_length_mm)
                            # world_co = img_to_world(xy)
                            line = (cls, *xywh, x, y, dis, conf) if save_conf else (cls, *xywh, x, y, dis)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp20/weights/best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/strawberries.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.65, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def mainOB(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))




shoulder_length = 13  # Example value, adjust according to your arm's dimensions
elbow_length = 23  # Example value, adjust according to your arm's dimensions

def calculate_angles(y, z, shoulder_length=shoulder_length, elbow_length=elbow_length):
    # Subtract 8cm from y to adjust for the joint being 8cm above the base
    y_adjusted = y - 8  # Adjust y to represent the distance from the joint

    # Calculate the shoulder angle
    #shoulder_angle = math.acos(((y_adjusted**2 + z**2) + shoulder_length ** 2 - elbow_length ** 2) / (2 * math.sqrt(y_adjusted**2 + z**2) * shoulder_length)) + math.atan(y_adjusted/z)
    shoulder_angle = 30

    # Calculate the elbow angle
    #elbow_angle = math.acos((shoulder_length**2 + elbow_length**2 - (y_adjusted**2 + z**2)) / (2 * elbow_length * shoulder_length))
    elbow_angle = 45

    return shoulder_angle, elbow_angle


def send_command(ser, base_angle, y, z, wrist_angle, grip_angle):
    # Calculate the shoulder and elbow angles based on the given y and z
    shoulder_angle, elbow_angle = calculate_angles(y, z, shoulder_length, elbow_length)
    shoulder_angle = 180 - shoulder_angle
    # Construct the command string
    command = f"MOVE,{base_angle},{shoulder_angle},{elbow_angle},{wrist_angle},{90},{grip_angle}\n"
    print(f"Sending: {command}")
    ser.write(command.encode('utf-8'))
    response = ser.readline().decode('utf-8').rstrip()
    print("Arduino:", response)
    time.sleep(1)  # Add a short delay to ensure the arm has time to move

def roboticArm(y, z):
    ser = serial.Serial('COM6', 9600, timeout=1)
    time.sleep(2)  # Wait for the connection to establish

    # Example command with y and z coordinates
    send_command(ser, 0, y, z, 90, 90)  # YoSu will need to replace y, z, wrist_angle, and grip_angle with actual values
    time.sleep(5)

    ser.close()


def webcontroller():
    opt = parse_opt()
    mainOB(opt)


if __name__ == '__main__':
    main()

    
# Iot client should always aim to be shutdown to avoid issues
