#!/bin/bash
cd "${0%/*}"
./bin/fbuild all
sudo chown root /tmp/Global\\FASTBuild-0x*
sudo ./bin/fbuild install
