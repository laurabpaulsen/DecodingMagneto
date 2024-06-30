
# start when PID 1348558 is done
#while [ $(ps -p 4109550 -o comm=) ]; do
#    sleep 60
#    echo "waiting for PID 4109550 to finish"
#done

python predict_session_day.py --trial_type "animate" --task "visual"
python predict_session_day.py --trial_type "inanimate" --task "visual"
python predict_session_number.py --trial_type "animate" --task "visual"
python predict_session_number.py --trial_type "inanimate" --task "visual"