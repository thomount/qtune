#!/usr/bin/env bash
# run_job.sh  selectedList.txt  queries_dir   output

avg_lat=0
avg_tps=0
count=0

printf "query\tlat(ms)\n" > $3

while read a ; do
  tmp=$(mysql -uroot -pDBMINDdbmind2020 imdbload < $2/$a | tail -n 1 )
  query=`echo $tmp | awk '{print $1}'`
  lat=`echo $tmp | awk '{print $2}'`
  tps=$(echo "scale=4; 60000 / $lat" | bc)

  echo $lat
  echo $tps

  avg_lat=$(echo "scale=4; $avg_lat + $lat / 1000" | bc)
  avg_tps=$(echo "scale=4; $avg_tps + $tps" | bc)
  ((count += 1))
  printf "$query\t$lat\n" >> $3

done < $1

((avg_lat /= count))
((avg_tps /= count))

printf "\navg_tps(txn/min): \t%5.4f\navg_lat(ms): \t%5.4f" $avg_tps $avg_lat >> $3
