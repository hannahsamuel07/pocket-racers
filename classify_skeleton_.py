import cv2
import numpy as np
import RPi.GPIO as GPIO
from threading import Thread
from multiprocessing import Process, Pipe
from queue import Queue
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

'''
DO NOT CHANGE THIS CLASS.
Parallelizes the image retrieval and processing across two cores on the Pi.
'''
class PiVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32):
        self.process = None
        self.resolution = resolution
        self.framerate = framerate


    def start(self):
        pipe_in, self.pipe_out = Pipe()
        # start the thread to read frames from the video stream
        self.process = Process(target=self.update, args=(pipe_in,), daemon=True)
        self.process.start()
        return self
    

    def update(self, pipe_in):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = self.resolution
        self.camera.framerate = self.framerate
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.rawCapture = PiRGBArray(self.camera, size=self.resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            pipe_in.send([self.frame])
            self.rawCapture.truncate(0)
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                self.process.join()
                return

    def read(self):
        # return the frame most recently read
        if self.pipe_out.poll():
            return self.pipe_out.recv()[0]
        else:
            return None


    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


print("[INFO] sampling MULTIPROCESSED frames from `picamera` module...")
vs = PiVideoStream(resolution=(640,480)).start()
time.sleep(2.0)

'''
DO NOT CHANGE THIS FUNCTION.

Annotates your filtered image with the values you calculate.

PARAMETERS:
img -               Your filtered BINARY image, converted to BGR or
                    RGB form using cv2.cvtColor().

contours -          The list of all contours in the image.

contour_index -     The index of the specific contour to annotate.

moment -            The coordinates of the moment of inertia of
                    the contour at `contour_index`. Represented as an
                    iterable with 2 elements (x, y).

midline -           The starting and ending points of the line that
                    divides the contour's bounding box in half,
                    horizontally. Represented as an iterable with 2
                    tuples, ( (sx,sy) , (ex,ey) ), where `sx` and `sy`
                    represent the starting point and `ex` and `ey` the
                    ending point.

instruction -       A string chosen from "left", "right", "straight", "stop",
                    or "idle".
'''
def part2_checkoff(img, contours, contour_index, moment, midline, instruction):
    img = cv2.drawContours(img, contours, contour_index, (0,0,255), 3)
    img = cv2.circle(img, (moment[0], moment[1]), 3, (0,255,0), 3)
    
    img = cv2.line(img,
                   midline[0],
                   midline[1],
                   (0, 0, 255),
                   3)
    
    img = cv2.putText(img,
                      instruction,
                      (50, 50),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      2,
                      (0,0,255),
                      2,
                      cv2.LINE_AA)

    return img, instruction
block_size = 9
# n = 3
def detect_shape(img):
    '''
    PART 1
    Isolate (but do not detect) the arrow/stop sign using image filtering techniques. 
    Return a mask that isolates a black shape on white paper

    Checkoffs: None for this part!
    '''

    adaptiveK = 1051
    adaptiveC = 5
    n = 7
    m = 7
    iterationsP = 2
    q = 2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    img = cv2.blur(img,(n,m)) #blur
    img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=adaptiveK, C=adaptiveC) # adaptive thresholding
    
    img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    img = cv2.dilate(img, np.ones((q,q), np.uint8), iterations=iterationsP) # dilate
    
    img = cv2.dilate(img, np.ones((q,q), np.uint8), iterations=iterationsP) # dilate
    img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    
    img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    # n = 15
    # m = 15
    # adaptiveK = 1551
    # adaptiveC = 25
    # iterationsP = 1
    # q = 2
    #img = color_img
    # color_code = cv2.COLOR_BGR2GRAY
    # img = cv2.cvtColor(img, color_code) # Changing it to grayscale
    # img = cv2.blur(img,(n,m)) #blur
    # img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=adaptiveK, C=adaptiveC) # adaptive thresholding
    # #ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # img = cv2.dilate(img, np.ones((q,q), np.uint8), iterations=iterationsP) # dilate
    # img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    
    # img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    # img = cv2.dilate(img, np.ones((q,q), np.uint8), iterations=iterationsP) # dilate


    ################################################################################

    # img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    # img = cv2.dilate(img, np.ones((q,q), np.uint8), iterations=iterationsP) # dilate
    
    # img = cv2.dilate(img, np.ones((q,q), np.uint8), iterations=iterationsP) # dilate
    # img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    
    # img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode

    
    
    ################################################################################




    #img = cv2.erode(img, np.ones((q,q), np.uint8), iterations=iterationsP) # erode
    
    # blur = cv2.GaussianBlur(img1,(n,n), 0)
    #img = cv2.adaptiveThreshold(blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=block_size,C=2)
    # bdp_color = cv2.SimpleBlobDetector_Params()

    # bdp_color.filterByColor = False

    # detector = cv2.SimpleBlobDetector_create(bdp_color)
    # img2 = detector.detect(img1)
    # im_with_keypoints = cv2.drawKeypoints(img1, img2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    '''
    END OF PART 1
    '''

    # Create the color image for annotating.
    formatted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Find contours in the filtered image.
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img #'idle'   # CHECK IDLE STATE MIGHT NEED TO RETURN SOMETHING ELSE
    
    '''
    PART 2
    1. Identify the contour with the largest area.
    2. Find the centroid of that contour.
    3. Determine whether the contour represents a stop sign or an arrow. If it's an
       arrow, determine which direction it's pointing.
    4. Set instruction to 'stop' 'idle' 'left' 'right' 'straight' depending on the output
    5. Use the part2_checkoff() helper function to produce the formatted image. See
       above for documentation on how to use it.

    Checkoffs: Send this formatted image to your leads in your team's Discord group chat.
    '''
    instruction = "idle"
    isArrow = False
    max_area = 0
    max_contour = None
    max_idx = 0

    for i in range (0, len(contours)):
        if (cv2.contourArea(contours[i])> max_area):
            max_area = cv2.contourArea(contours[i])
            max_contour = contours[i]
            max_idx = i
    
    # for contour in contours:
    #     if (cv2.contourArea[contour] > max_area):
    #         max_area = cv2.contourArea[contour]
    #         max_contour = contour
    x, y, w, h = cv2.boundingRect(contours[max_idx])

    try:
        moments = cv2.moments(max_contour)
        mx = int(moments["m10"]/ moments["m00"])
        my = int(moments["m01"]/ moments["m00"])
        centroid = (mx, my)

    except ZeroDivisionError:
        print("Cannot divide by zero.")
    
    isConvex = cv2.isContourConvex(max_contour)
    if isConvex :
        instruction = "stop"
    # else :
    #     isArrow = True

        # epsilon = abs((x - mx) / 10)
    epsilon_1 = w / 10

    if (abs(w - h) < epsilon_1):
        instruction = "stop"

    else:
        isArrow = True

    #print("THis is te output of isConvex: ", isArrow, " also: ", abs(w - h))
    
    midline_top = (int((x + w) / 2), int(y))
    midline_bottom = (int((x + w) / 2), int(y + h))

    midline = (midline_top, midline_bottom)

    epsilon_2 = w / 5

    mid_y = abs(y + h / 2)

    epsilon_3 = h / 20
    #print("abs(my - mid_y): ", abs(my - mid_y), " also ", mid_y, " and ", my)
    if(isArrow):
        if (abs(my - mid_y) <= epsilon_3):
            if ((abs(x - mx) < (abs(mx - (x + w))))):
                    instruction = 'left'
            elif ((abs(x - mx) > (abs(mx - (x + w))))):
                    instruction = 'right'
        elif ((abs(x - mx) - abs((x + w) - mx) )<= epsilon_2):
            if (abs(y - my) < abs(my - (y + h))):
                instruction = 'straight'
            else: 
                instruction = 'idle'
        else:
            instruction = 'idle'
        # if ((abs(x - mx) - abs((x + w) - mx) )<= epsilon_2): #Vertical
        #     if ((y - my) < (my - (y - h))):
        #         instruction = 'straight'
        # else:
        #     if ((abs(x - mx) < (abs(mx - (x + w))))):
        #         instruction = 'left'
        #     else:
        #         instruction = 'right'
    '''
    END OF PART 2
    '''
    if (cv2.contourArea(max_contour)/cv2.contourArea(cv2.convexHull(max_contour)) < .475):
        instruction = 'idle'
    return part2_checkoff(formatted_img, contours, max_idx, centroid, midline, instruction)

    #return img

'''
PART 3
0. Before doing any of the following, arm your ESC by following the instructions in the
   spec. You only have to do this once. Than the range will be remembered by the ESC
1. Set up two GPIO pins of your choice, one for the ESC and one for the Servo.
   IMPORTANT: Make sure your chosen pins aren't reserved for something else! See pinout.xyz
   for more details.
2. Start each pin with its respective "neutral" pwm signal. This should be around 8% for both.
   The servo may be slightly off center. Fix this by readjusting the arm of the servo (unscrew it,
   set the servo to neutral, make the wheel point straight, then reattach the arm). The arm may still
   not be perfectly alighned so use the manual_pwm.py program to determine your Servo's best neutral
   position.
3. Start the motor at the full-forward position (duty cycle = 5.7).

NOTE: If you change the variable names pwm_m and pwm_s, you'll also need to update the
      cleanup code at the bottom of this skeleton.

Checkoffs: None for this part!
'''

pwm_m = None
pwm_s = None
print("started!")

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

GPIO.setup(37, GPIO.OUT)
#Mode (High is Pi Control, Low is RC)


GPIO.output(37, GPIO.HIGH)


GPIO.setup(11, GPIO.OUT)
p_left = GPIO.PWM(11, 60)
p_left.start(8)

GPIO.setup(13, GPIO.OUT)
p_right = GPIO.PWM(13, 60)
p_right.start(8)
# p.ChangeDutyCycle(dc)
'''
END OF PART 3
'''

'''
PART 4
1. 
'''

frame_count = 0
left_count = 0
right_count = 0
last_instruction = None

try:
    while True:
        if vs.pipe_out.poll():
            result = vs.read()
            img = cv2.rotate(result, cv2.ROTATE_180)
            scaleX = 0.5
            scaleY = 0.5
            new_dims = (int(img.shape[1] * scaleX), int(img.shape[0] * scaleY))
            img = cv2.resize(img, new_dims)
            frame_count += 1
            if frame_count == 1:
                print(img.shape)

            instruction, last_instruction = detect_shape(img)

            cv2.imshow("Capture", instruction)
            # print(detect_shape(img))
            '''
            PART 4
            1. Figure out the values of your motor and Servo PWMs for each instruction
               from `detect_shape()`.
            2. Assign those values as appropriate to the motor and Servo pins. Remember
               that an instruction of "idle" should leave the car's behavior UNCHANGED.

            Checkoffs: Show the leads your working car!
            '''

            #last_instruction = instruction
            # print( last_instruction == "straight")
            if (last_instruction == "idle"):
                True
            elif (last_instruction == "left"):
                p_left.ChangeDutyCycle(0)
                p_right.ChangeDutyCycle(15)
            elif (last_instruction == "right"):
                p_left.ChangeDutyCycle(5.7)
                p_right.ChangeDutyCycle(0)
            elif (last_instruction == "stop"):
                p_left.ChangeDutyCycle(0)
                p_right.ChangeDutyCycle(0)
            elif (last_instruction == "straight"):
                p_left.ChangeDutyCycle(5.7)
                p_right.ChangeDutyCycle(15)
            
            '''
            END OF PART 4
            '''

            k = cv2.waitKey(3)
            if k == ord('q'):
                # If you press 'q' in the OpenCV window, the program will stop running.
                break
            elif k == ord('p'):
                # If you press 'p', the camera feed will be paused until you press
                # <Enter> in the terminal.
                input()
except KeyboardInterrupt:
    pass

# Clean-up: stop running the camera and close any OpenCV windows
pwm_m.stop()
pwm_s.stop()
GPIO.cleanup()
cv2.destroyAllWindows()
vs.stop()