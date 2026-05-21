### Get a password
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword

# Get logs
docker exec -it jenkins bash
cd /var/jenkins_home/workspace/Train
cat logs.txt