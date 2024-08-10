git clone https://github.com/Fotoblysk/avg-transfer-rl
echo $MY_RUNNER_ID > runner_id
cd avg-transfer-rl
git checkout cloud_sync
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
./bash_runnup
tar -zcvf /root/results.tar.gz ./results
touch /root/jobs_done