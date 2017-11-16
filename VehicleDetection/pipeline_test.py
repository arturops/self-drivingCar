from moviepy.editor import VideoFileClip
from IPython.display import HTML
from car_finder import *

def process_image(image):
    """ This processes through everything above.
    Will return the image with other vehicles shown boxed in blue,
    our own car position, lane curvature, and lane lines drawn.
    """
    # Vehicle Detection
    result = car_detection(image)
    
    return result

# Convert to video
# vid_output is where the image will be saved to
video_output = './output_videos/project_video_output.mp4'

# The file referenced in clip1 is the original video before anything has been done to it
clip1 = VideoFileClip("./test_videos/project_video.mp4")

# NOTE: this function expects color images
vid_clip = clip1.fl_image(process_image) 
vid_clip.write_videofile(video_output, audio=False)