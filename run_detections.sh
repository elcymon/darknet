# if (($1)); then
#     cd /mnt
#     echo "HPC"
# else
#     echo "local"
# fi


#python3 object_detection_yolo.py --video=videos/20190111GOPR9027.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=0.0 --imgSize=416
#python3 object_detection_yolo.py --video=videos/20190111GOPR9027.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=1.0 --imgSize=416
#python3 object_detection_yolo.py --video=videos/20190111GOPR9027qtr.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=0.0 --imgSize=416
#python3 object_detection_yolo.py --video=videos/20190111GOPR9027qtr.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=1.0 --imgSize=416

python3 object_detection_yolo.py --video=videos/20190111GOPR9027.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=0.0 --imgSize=216