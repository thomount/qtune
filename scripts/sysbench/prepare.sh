#!/usr/bin/env bash

script_path="/web/dbmind-bg/AutoTuner/scripts/"

if [ ${1} == "read" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ ${1} == "write" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi

sysbench ${run_script} \
	--mysql-host=$2 \
	--mysql-port=$3 \
	--mysql-user=dbmind \
	--mysql-password=$4 \
	--mysql-db=sysbench_test \
	--db-driver=mysql \
	--tables=2 \
	--table-size=4000000 \
	--report-interval=10 \
	--threads=3 \
	--time=60 \
	prepare
