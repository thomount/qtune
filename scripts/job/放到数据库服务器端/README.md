# join-order-benchmark-mysql
join-order-benchmark(JOB) for mysql

# Prepare data
download data files from http://homepages.cwi.nl/~boncz/job/imdb.tgz and extract csv files
```
mkdir job; cd job;
mkdir imdb-2014-csv-mysql
tar zxvf imdb.tgz 
mv *.csv imdb-2014-csv-mysql
```

# Load data
add a new line `sql_mode=NO_ENGINE_SUBSTITUTION` to `my.cnf` or `my.ini`, and restart mysqld

then run the followings:
```
mysql -uroot -S$MYSQL_SOCK -e "drop database if exists imdbload"
mysql -uroot -S$MYSQL_SOCK -e "create database imdbload"
mysql -uroot -S$MYSQL_SOCK  imdbload < schema.sql

cat table_list.txt | while read a ; do 
echo "LOAD DATA INFILE '`pwd`/imdb-2014-csv-mysql/$a.csv' INTO TABLE $a FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"';"
done > load-data.sql

mysql -uroot -S$MYSQL_SOCK imdbload < load-data.sql
mysql -uroot -S$MYSQL_SOCK imdbload < fkindexes.sql

cat table_list.txt | while read a ; do 
  echo "analyze table $a;"
done > analyze-tables.sql

mysql -uroot -S$MYSQL_SOCK imdbload < analyze-tables.sql
```

# Run workload
to run all 113 queries:
```
chmod u+x run_job.sh
./run_job.sh all_queries.txt queries-mysql/ output.res
```
you can also specify queries and execution order in a txt file, like `selected.txt`:
```
./run_job.sh selected.txt queries-mysql/ output.res
```
# Output
output is like the followings:
```
query   lat(ms)
3a      241.113000000000000000000000000
13d     12175.714000000000000000000000000
17a     8141.935000000000000000000000000
11d     498.170000000000000000000000000
25c     38135.314000000000000000000000000
2d      578.571000000000000000000000000
31a     574.830000000000000000000000000
4a      3771.808000000000000000000000000
23b     537.806000000000000000000000000
12b     4400.070000000000000000000000000
20b     2335.184000000000000000000000000
3b      130.435000000000000000000000000
33b     9.086000000000000000000000000
28a     3147.823000000000000000000000000

avg_throughput(txn/min):        560.0476
avg_latency(ms):        5334.1327
```
