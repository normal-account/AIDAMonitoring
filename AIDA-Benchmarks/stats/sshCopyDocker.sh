ssh cbilod2@mimi.cs.mcgill.ca -i ~/.ssh/id_rsa "./copy_docker.sh"
sftp -b sftp_batch.txt cbilod2@mimi.cs.mcgill.ca
