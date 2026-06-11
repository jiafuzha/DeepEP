# build docker if no yet.
# docker build -t jiafuzha-ishmem-ibgda:latest .
cd tests/docker-2node                                                                                                                                   ┃
docker rm -f deepep-node0 deepep-node1                                                                                                                  ┃
NUM_PROCESSES=2 NUM_TOKENS=32 HIDDEN=1024 NUM_TOPK=2 NUM_EXPERTS=8 TIMEOUT_SEC=180 ./run.sh