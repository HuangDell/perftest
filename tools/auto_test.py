import os
import subprocess
import time
import socket
from datetime import datetime

# 设置参数
MAX_RESTARTS = 30
REPEAT_COUNT = 10000  # -n 参数的值
PACKET_SIZE = 65536

QP_COUNT = 1
restart_count = 0

# 文件名参数
ft_value = 5000 # 可以根据需要修改
thre_value = 0  # 可以根据需要修改
# 可以根据需要修改  v4 for fct test
# v5 for qp = 2
# v6 for qp = 3
# v7 for qp = 4
# v8 for qp = 6
# v9 for qp = 8
# ali for alistorage

version = "AliStorage2019"  
cdf_file_list = ['AliStorage2019','FbHdp2015','GoogleRPC2008','Solar2022']


RESOURCES_DIR = "./out/prototype/" +version 
FILENAME = f"prototype_ft_{ft_value}_thre_{thre_value}_{version}.log"
LOG_FILE = os.path.join(RESOURCES_DIR, FILENAME)  
MODE = "./ib_send_bw"
CDF_FILE = ""

cmd = ["sudo", MODE]
wait_time = None

# 获取主机名
hostname = socket.gethostname()

# 创建resources目录（如果不存在）  
os.makedirs(RESOURCES_DIR, exist_ok=True)  

# 删除已存在的日志文件并创建新的
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
open(LOG_FILE, 'a').close()

print(f"Starting application monitor... Maximum restarts: {MAX_RESTARTS}")
print(f"Current hostname: {hostname}")
print(f"Log file: {LOG_FILE}")


while restart_count < MAX_RESTARTS:
    # 根据主机名构建不同的命令
    if hostname == "FNIL-2022DEC-GPU-7":
        cmd += ["-d", "mlx5_1", "-n", str(REPEAT_COUNT),"-s", str(PACKET_SIZE),"-q",str(QP_COUNT)]
        wait_time = 1

    elif hostname == "FNIL-2022DEC-GPU-8":
        cmd += ["10.10.10.4", "-n", str(REPEAT_COUNT),"-s", str(PACKET_SIZE),"-q",str(QP_COUNT)]
        wait_time = 3
    else:
        message = f"Unsupported hostname: {hostname}"
        print(message)
        with open(LOG_FILE, 'a') as log:
            log.write(message + '\n')
        break
    if version in cdf_file_list:
        cmd += ["--use-cdf", f"./resources/{version}.txt"]
        
    # 打开日志文件以追加模式
    with open(LOG_FILE, 'a') as log:
        # 记录当前执行的命令
        log.write(f"\nExecuting command: {' '.join(cmd)} at {datetime.now()}\n")
        
        # 运行命令并重定向输出到日志文件
        process = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    # 增加重启计数
    restart_count += 1
    
    # 如果达到最大重启次数，退出循环
    if restart_count == MAX_RESTARTS:
        message = f"Maximum restart count ({MAX_RESTARTS}) reached at {datetime.now()}. Stopping monitor..."
        print(message)
        with open(LOG_FILE, 'a') as log:
            log.write(message + '\n')
        break
    
    # 应用退出后，输出重启信息到日志
    message = f"Application exited at {datetime.now()}. Restart count: {restart_count}/{MAX_RESTARTS}. Waiting {wait_time} second before restart..."
    print(message)
    with open(LOG_FILE, 'a') as log:
        log.write(message + '\n')
    
    time.sleep(wait_time)