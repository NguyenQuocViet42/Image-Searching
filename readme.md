docker run --name image_searching -e MYSQL_ROOT_PASSWORD=123 -e MYSQL_DATABASE=image_searching -d mysql:latest

docker exec -it image_searching mysql -uroot -p123

docker exec image_searching mysqldump -uroot -p123 image_searching > mysql/mydatabase_backup.sql

docker exec -i image_searching mysql -uroot -p123 image_searching < mysql/mydatabase_backup.sql

SHOW DATABASES;
USE image_searching;
SHOW TABLES;