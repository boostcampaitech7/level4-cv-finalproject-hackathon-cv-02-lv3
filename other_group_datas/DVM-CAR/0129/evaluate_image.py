import ollama
import glob
import os
import re
import pandas as pd
from tqdm import tqdm
import signal, time

def timeout_handler(signum, frame):
    print("timeout")


def validate_path(path):
    validate_path = ""
    for c in path:
        if c in "\:*?<>|":
            validate_path += "_"
        else:
            validate_path += c

    return validate_path

def chat(model, instruction, image_path=None):
    messages = {'role': 'user', 'content': instruction}
    if image_path is not None:
        messages['images'] = [image_path]

    messages = [messages] # List로 변환 
    response = ollama.chat(model=model, messages=messages)
    content = response['message']['content']
    return content

def find_score_from_response(pattern, response):
    matches = re.findall(pattern, response)
    score = -1

    if matches: # 일치하는 패턴 찾음
        try:
            score = float(matches[-1]) # 마지막 매칭 결과를 score로 반영
        except:
            pass

    return score

def get_score_from_chat(model, instruction, image_path=None, pattern = r"\d\.\d{4}", timeout=30):
    n_max_trial = 20 # score을 찾지 못했을 때 최대 시도 횟수
    score = -1

    for _ in range(n_max_trial):
        signal.signal(signal.SIGALRM, timeout_handler) # 시간 제한
        signal.alarm(timeout)
        
        try: 
            response = chat(model, instruction, image_path)
            score = find_score_from_response(pattern, response)
        finally:
            signal.alarm(0)
            
        if (0 <= score and score <= 1): # 정상적인 score 값이면 종료
            break
    
    response = response.replace("\n", "") # 엔터 제거
    response = response.replace("\r", "") # 엔터 제거
    return score, response


def write_file(path, content):
    with open(path, 'a') as f:
        f.write(content)
        f.flush()


if __name__ == '__main__':
    model = 'minicpm-v'
    image_dir_path = '../data/confirmed_fronts'
    save_dir_path = f'result/{model}/stylishness'
    years = list(range(2018, 1990, -1))

    instruction = (
        "Please disregard any previous instructions. "
        "You are a professional car designer tasked with evaluating the stylishness "
        "of cars using numerical scores. Assess the stylishness of the car shown in the images "
        "below by assigning a numerical score between 0 and 1, where 0 represents "
        "'not stylish at all' and 1 signifies 'extremely stylish.' Provide the score "
        "with four decimal places (for example, 0.1322)."
        )

    save_dir_path = validate_path(save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True) # 폴더 생성

    for year in years:
        print(f"***** {year}년 확인 중 *****")
        image_paths = glob.glob(image_dir_path + f'/*/{year}/*', recursive=True)
        save_path = os.path.join(save_dir_path, f'{year}.csv')
        save_instruction_path = save_path.replace('csv', 'txt')

        df = pd.DataFrame(columns=['image_path', 'score', 'response'])

        if os.path.exists(save_path): # 이미 존재하는 경로이면 무시
            continue
        write_file(save_path, "")

        for image_path in tqdm(image_paths, dynamic_ncols=False):
            rel_image_path = image_path.replace(image_dir_path + '/', "")
            score, response = get_score_from_chat(model=model, instruction=instruction, image_path=image_path)
            df.loc[len(df)] = [rel_image_path, score, response]
            tqdm.write(f"{rel_image_path} - score: {score:.4f}")

        # 파일 저장
        df.to_csv(save_path, index=False)
        write_file(save_instruction_path, instruction)
        
        
