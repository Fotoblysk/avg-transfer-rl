sleep 60

while true; do
    git clone https://github.com/Fotoblysk/avg-transfer-rl

    if [ $? -eq 0 ]; then
        echo "Git clone succeeded!"
        break
    else
        echo "Git clone failed. Retrying in 5 seconds..."
        sleep 5
    fi
done

echo $MY_RUNNER_ID > runner_id
cd avg-transfer-rl

python3 -m venv venv
source venv/bin/activate

while true; do
    pip3 install -r requirements.txt

    if [ $? -eq 0 ]; then
        echo "pip succeeded!"
        break
    else
        echo "pip failed. Retrying in 5 seconds..."
        sleep 5
    fi
done

./bash_runnup

while true; do
    tar -zcvf /root/results.tar.gz ./results

    if [ $? -eq 0 ]; then
        echo "tar succeeded!"
        break
    else
        echo "tar failed. Retrying in 5 seconds..."
        sleep 5
    fi
done
touch /root/jobs_done