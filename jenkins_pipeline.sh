#№1. download
python3 -m venv ./my_env #создать виртуальное окружение в папку 
. ./my_env/bin/activate   #активировать виртуальное окружение
pip3 install -r requirements.txt    #установить пакеты python3
python3 download.py    #запустить python3 script
#-----------------------

#№2. train_model 
echo "Start train model"
. ./my_env/bin/activate   #активировать виртуальное окружение
python3 train_model.py > logs.txt
#------------------------

# #3. deploy 
# cd /var/lib/jenkins/workspace/download/
# . ./my_env/bin/activate   #активировать виртуальное окружение
# cd ./MLOPS/lab3		   #перейти в директорию ./MLOPS/lab3
# export BUILD_ID=dontKillMe            #параметры для jenkins чтобы не убивать фоновый процесс для mlflow сервиса
# export JENKINS_NODE_COOKIE=dontKillMe #параметры для jenkins чтобы не убивать фоновый процесс для mlflow сервиса
# path_model=$(cat best_model.txt) #прочитать путь из файла в bash переменную 
# mlflow models serve -m $path_model -p 5003 --no-conda & #запуск mlflow сервиса на порту 5003 в фоновом режиме
# #------------------------

# #4. healthy (status service)
# curl http://127.0.0.1:5003/invocations -H"Content-Type:application/json"  --data '{"inputs": [[ -1.275938045, -1.2340347 , -1.41327673,  0.76150439,  2.20097247, -0.410937195,  0.58931542,  0.1135538,  0.58931542]]}'


#Pipeline - для объедения задач в последовательный конвеер
pipeline {
    agent any

    stages {
        stage('Download') {
            steps {
                
                build job: 'download'
            }
        }
        
        stage ('Train') {
            
            steps {
                build job: 'train_model'
            
            }
        }
        
        # stage ('Deploy') {
        #     steps {
        #         build job: 'deploy'
        #     }
        # }
        
        # stage ('Status') {
        #     steps {
        #         build job: 'healthy'
        #     }
        # }
    }
}