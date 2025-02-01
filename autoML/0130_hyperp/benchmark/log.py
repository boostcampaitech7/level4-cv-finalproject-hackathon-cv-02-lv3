from datetime import datetime
import os

class do_log:
    def __init__(self):
        now = datetime.now()
        time_string = now.strftime("%y%m%d_%H%M%S")
        py_dir_path = os.path.dirname(os.path.abspath(__file__))

        self.log_dir_path = os.path.join(py_dir_path, 'log')
        self.log_path = os.path.join(self.log_dir_path, f"{time_string}.txt")
        os.makedirs(self.log_dir_path, exist_ok=True)


    def log_dicts(self, dicts, message=""):
        log = []
        for k, v in dicts.items():
            if isinstance(v, float):
                log.append(f'{k}: {v:.4f}')
            else:
                log.append(f'{k}: {v}')
        
        log = ', '.join(log)
        if len(message):
            log = f'{message} - {log}'
        self.log(log)
    

    def log(self, message):
        """
        log 기록

        Args:
            message (str): log 메세지
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{now}] {message}"
        print(log_message) # 로그 출력 

        # 로그 저장
        with open(self.log_path, 'a') as file:
            file.write(log_message + "\n") 
            file.flush()