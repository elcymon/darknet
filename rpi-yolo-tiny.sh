module load ffmpeg
module load opencv
module load cuda

#thresh=$1 # threshold for detection
threshstr=$(echo $thresh | tr '.' 'p') # replace decimal with p
#fname=$2 #video filename without extension
#ntwk=$3 #yolo network to use
#iteration=$4 #yolo weight iteration to work on
output=tiny-10000  #"$fname"-"$ntwk"_"$iteration"-"$threshstr"-v2

#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -prefix output0p01 videos/20190111GOPR9025.MP4 -thresh 0.01
#./darknet detector demo litter/litter.data litter/yolov3-litter.cfg backup/yolov3-litter_900.weights -prefix outputYOLOlitter videos/20190111GOPR9025.MP4 -thresh 0.1
#./darknet detector demo litter/litter.data litter/"$ntwk".cfg backup/"$ntwk"_"$iteration".weights -prefix $output videos/"$fname".MP4 -thresh $thresh

ffmpeg -framerate 50 -i "$output"_%08d.jpg rpi-yolo-"$output".mp4
rm "$output"*.jpg


