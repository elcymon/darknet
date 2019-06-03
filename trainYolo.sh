module load opencv
module load cuda

# train
# ./darknet detector train litter/litter.data litter/yolov3-tiny-litter.cfg darknet53.conv.74
# ./darknet detector train litter/litter.data litter/yolov3-litter.cfg darknet53.conv.74
# resume
# ./darknet detector train litter/litter.data litter/yolov3-tiny-litter.cfg backup/yolov3-tiny-litter.backup 
./darknet detector train litter/litter.data litter/yolov3-litter.cfg backup/yolov3-litter.backup 

