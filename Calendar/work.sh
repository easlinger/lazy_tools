#!/bin/bash
# Display hours worked this or last week
# Use argument last for last week

cd
cd /home/ena/scripts/Python/Calendar

python /home/ena/scripts/Python/Calendar/work.py $1 $2 2>&1
