import json
import pandas as pd
import sys

class_names = [
    'Traveling',
    'Lifting Brick',
    'Lifting Rebar',
    'Measuring Rebar',
    'Tying Rebar',
    'Hammering',
    'Drilling',
    'Idle'
]

correct_pred = {classname: 0 for classname in class_names}
total_pred = {classname: 0 for classname in class_names} 


if __name__ == '__main__':
    with open('results/val.json', 'r') as f:
        results = json.load(f)

    excel = pd.read_excel('hongkong/train_data.xlsx')
    excel = excel[['youtube_id', 'label']]
    df = excel.drop_duplicates()
    gts = {}
    for v in df.iterrows():
        gts[v[1]['youtube_id']] = v[1]['label']

    for k, v in results['results'].items():
        gt = gts[k[:8]]
        if gt == v[0]['label']:
            correct_pred[gt] += 1
        total_pred[gt] += 1

    cnt = 0
    total = 0

    for k, _ in correct_pred.items():
        print(f'{k} class is {correct_pred[k]/total_pred[k] * 100}% accuracy')
        cnt += correct_pred[k]
        total += total_pred[k]

    print()
    print(f'All class is {cnt/total*100}% accuracy')
    print()
    


    

    