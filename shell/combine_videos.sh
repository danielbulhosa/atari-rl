# Use to combine all episode videos into a single one, delete videos
ls *.mp4 | while read -r a; do echo "file \"${a}\""; done >> video_list.txt
ffmpeg -f concat -safe 0 -i video_list.txt -codec copy output.mp4
rm -- !(/output.mp4)
