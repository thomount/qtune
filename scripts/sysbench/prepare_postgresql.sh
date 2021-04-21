#!/usr/bin/env bash

script_path="/web/dbmind-bg/AutoTuner/scripts/"

run_script=${script_path}"oltp_read_only.lua"

sysbench ${run_script} \
	--db-driver=pgsql \
	--pgsql-host=localhost \
	--pgsql-port=5432 \
	--pgsql-user=dbmind \
	--pgsql-password=dbmind2020 \
	--pgsql-db=dbmind \
	--tables=2 \
	--table-size=4000000 \
	--report-interval=10 \
	--threads=3 \
	--time=60 \
	prepare
