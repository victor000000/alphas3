import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth
import tqdm
import logging
import time
import os
import ast
from datetime import datetime
from pytz import timezone
import csv
import sys


class AlphaSimulator:
    def __init__(self, max_concurrent, username, password, alpha_list_filepath, batch_number_for_every_queue):
        estern = timezone('US/Eastern')
        fmt = '%Y-%m-%d %H:%M:%S %Z%z'
        loc_dt = datetime.now(estern)
        print("Local time:", loc_dt.strftime(fmt))

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('day4_simulations.log'), logging.StreamHandler()])
        self.fail_alphas = 'failec_alphas.csv'
        self.simulated_alphas = f'simulated_alphas_{loc_dt.strftime(fmt)}.csv'
        self.max_concurrent = max_concurrent
        self.username = username
        self.password = password
        self.alpha_list_filepath = alpha_list_filepath
        self.batch_number_for_every_queue = batch_number_for_every_queue
        self.active_simulations = []
        self.session = self.sign_in(username, password)
        self.sim_queue_ls = []


    def sign_in(self, username, password):
        s = requests.Session()
        s.auth = (username, password)
        count = 0
        count_limit = 30
        while True:
            try:
                response = s.post("https://api.worldquantbrain.com/authentication")
                response.raise_for_status()
                break
            except:
                count += 1
                logging.error("Connection doin, trying to login again...") 
                time.sleep(15)
                if count > count_limit:
                    logging.error(f"{username} Failed too many times, returning None.")
                    return None
        logging. info("Login to BRAIN successfully.")
        return s

    def read_alphas_from_csv_in_batches(self, batch_size=50):
        """
        从alpha待处理CSV文件中批量读取数据
        :param batch_size: 每次读取的批次大小，默认50
        :return: 读取到的alpha列表（字典格式，每行为一个alpha）
        """
        alphas = []  # 存储当前批次读取的alpha
        # 临时文件路径（用于覆盖原文件时避免数据丢失）
        temp_file_name = f"{self.alpha_list_filepath}.tmp"

        # 步骤1：读取原文件，提取batch_size条数据，剩余数据写入临时文件
        with open(self.alpha_list_filepath, "r", encoding="utf-8") as file, \
             open(temp_file_name, "w", newline="", encoding="utf-8") as temp_file:
            
            # 初始化CSV读取器（获取表头）
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames  # 保存CSV表头（确保写入时格式一致）
            
            # 初始化CSV写入器（用于写剩余数据到临时文件）
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)  # 修复：DictWriter首字母大写，fieldnames赋值格式
            writer.writeheader()  # 写入表头

            # 步骤2：提取batch_size条数据到alphas列表
            for _ in range(batch_size):
                try:
                    row = next(reader)  # 逐行读取数据
                    
                    # 处理settings字段：若为字符串则转字典，若为字典则跳过
                    if "settings" in row:
                        if isinstance(row["settings"], str):
                            try:
                                row["settings"] = ast.literal_eval(row["settings"].replace("true", "True").replace("false", "False"))  # 字符串转字典
                            except (ValueError, SyntaxError) as e:
                                # 修复：字符串插值格式错误，添加异常信息
                                print(f"Error evaluating settings: {row['settings']}, Exception: {e}")
                        elif isinstance(row["settings"], dict):
                            pass  # 已为字典，无需处理
                        else:
                            # 修复：类型判断的打印格式错误
                            print(f"Unexpected type for settings: {type(row['settings'])}")
                    
                    alphas.append(row)  # 将处理后的行加入当前批次

                except StopIteration:
                    # 原文件已无数据，终止循环
                    break

            # 步骤3：将剩余未读取的数据写入临时文件（覆盖原文件前的备份逻辑）
            for remaining_row in reader:  # 修复：变量名"remaining row"→"remaining_row"
                writer.writerow(remaining_row)

        # 步骤4：用临时文件覆盖原文件（实现"取出后覆盖原列表"的需求）
        os.replace(temp_file_name, self.alpha_list_filepath)

        # 步骤5：将当前批次的alphas写入sim_queue.csv（用于监控排队数量）
        if alphas:  # 仅当有数据时才写入
            # 修复：文件名"sim_queue.cs"→"sim_queue.csv"，补充编码格式
            with open("sim_queue.csv", "w", newline="", encoding="utf-8") as file:
                # 用当前批次数据的表头初始化写入器
                writer = csv.DictWriter(file, fieldnames=alphas[0].keys())
                # 若文件为空（首次写入），先写表头
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerows(alphas)  # 写入当前批次所有数据

        # 步骤6：返回当前批次的alpha列表
        return alphas
    
    def submit_simulation(self, alpha):
        count = 0
        while True:
            try:
                response = self.session.post("https://api.worldquantbrain.com/simulations", json=alpha)
                response.raise_for_status()
                return response.headers['Location']
            except requests.exceptions.RequestException as e:
                count += 1
                logging.error(f"Error submitting simulation for alpha {alpha}: {e}. Retrying...")
                time.sleep(10)
                if count > 35:
                    try:
                        self.session = self.sign_in(self.username, self.password)
                        logging.info("Re-login successful.")
                    except Exception as e:
                        logging.error(f"Re-login failed: {e}")
                    break
        logging.error(f"Failed to submit simulation for alpha {alpha} after {count} attempts.")
        with open(self.fail_alphas, "a", newline="", encoding="utf-8") as fail_file:
            writer = csv.DictWriter(fail_file, fieldnames=alpha.keys())
            if fail_file.tell() == 0:
                writer.writeheader()
            writer.writerow(alpha)
        return None
    
    def load_new_alpha_and_simulate(self):
        if len(self.sim_queue_ls) < 1:
            self.sim_queue_ls = self.read_alphas_from_csv_in_batches(self.batch_number_for_every_queue)
        
        if len(self.active_simulations) >= self.max_concurrent:
            logging.info(f"Max concurrent simulations reached ({self.max_concurrent}). Waiting 13 seconds")
            time.sleep(13)
            return
        
        logging.info("Loading new alpha...")
        try:
            alpha = self.sim_queue_ls.pop(0)
            logging.info(f"Starting simulation for alpha: {alpha['regular']} with settings: {alpha['settings']}")
            location_url = self.submit_simulation(alpha)
            if location_url:
                self.active_simulations.append(location_url)
        except IndexError:
            logging.info("No more alphas available in the queue.")

    def check_simulation_progress(self, simulation_progress_url):
        try:
            simulation_progress = self.session.get(simulation_progress_url)
            simulation_progress.raise_for_status()
            if simulation_progress.headers.get("Retry-After", 0) == 0:
                alpha_id = simulation_progress.json().get("alpha")
                if alpha_id:
                    alpha_response = self.session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}")
                    alpha_response.raise_for_status()
                    return alpha_response.json()
                else:
                    return simulation_progress.json()
            else:
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching simulation progress: {e}")
            self.session = self.sign_in(self.username, self.password)
            return None
    
    def check_simulation_status(self):
        count = 0
        if len(self.active_simulations) == 0:
            logging.info("No one is in active simulation now")
            return None
        
        for sim_url in self.active_simulations:
            sim_progress = self.check_simulation_progress(sim_url)
            if sim_progress is None:
                count += 1
                continue
                
            alpha_id = sim_progress.get("id")
            status = sim_progress.get("status")
            
            logging.info(f"Alpha id: {alpha_id} ended with status: {status}. Removing from active list.")
            self.active_simulations.remove(sim_url)
            with open(self.simulated_alphas, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=sim_progress.keys())
                writer.writerow(sim_progress)
        
        logging.info(f"Total {count} simulations are in process for account {self.username}.")

    def manage_simulations(self):
        if not self.session:
            logging.error("Failed to sign in. Exiting..")
            return
        
        while True:
            self.check_simulation_status()
            self.load_new_alpha_and_simulate()
            time.sleep(2)

# Example usage
alpha_list_file_path = "alpha_pending_simulation_list_2025-09-25 07:50:54 EDT-0400.csv"  # replace with your actual file path
# Load credentials
with open(expanduser("brain_credentials.txt")) as f:
    credentials = json.load(f)
# Extract username and password from the list
username, password = credentials
max_concurrent = 3
batch_number = 10

simulator = AlphaSimulator(max_concurrent, username, password, alpha_list_file_path, batch_number)
simulator.manage_simulations()