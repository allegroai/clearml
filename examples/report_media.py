# TRAINS - Example reporting video or audio links/file
#
import os
from trains import Task, Logger


task = Task.init(project_name="examples", task_name="Reporting audio and video")

# report an already uploaded video media (url)
Logger.current_logger().report_media(
    'video', 'big bunny', iteration=1,
    url='https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4')

# report an already uploaded audio media (url)
Logger.current_logger().report_media(
    'audio', 'pink panther', iteration=1,
    url='https://www2.cs.uic.edu/~i101/SoundFiles/PinkPanther30.wav')

# report local media file
Logger.current_logger().report_media(
    'audio', 'tada', iteration=1,
    local_path=os.path.join('samples', 'sample.mp3'))
