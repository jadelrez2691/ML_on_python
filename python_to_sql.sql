CREATE DATABASE pydb;
USE pydb;

#create user  and set password
CREATE USER 'pydbuser'@'localhost' IDENTIFIED BY 'pydbpwb123';

#grant select privilege to the user
GRANT SELECT ON pydb.* To 'pydbuser'@'localhost';

select * from baseball;



