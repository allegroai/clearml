# ClearML - Example reporting video or audio links/file
#
import os
from clearml import Task, Logger


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="examples", task_name="Audio and video reporting")

print('reporting audio and video samples to the debug samples section')

# report video, an already uploaded video media (url)
Logger.current_logger().report_media(
    'video', 'big bunny', iteration=1,
    url='https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4')

#  report audio, report an already uploaded audio media (url)
Logger.current_logger().report_media(
    'audio', 'pink panther', iteration=1,
    url='https://www2.cs.uic.edu/~i101/SoundFiles/PinkPanther30.wav')

#  report audio, report local media audio file
Logger.current_logger().report_media(
    'audio', 'tada', iteration=1,
    local_path=os.path.join('data_samples', 'sample.mp3'))
