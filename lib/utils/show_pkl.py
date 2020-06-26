import pickle

path = '/Users/hby/Documents/XJTU/华为大数据比赛/初赛/Demo/truth_anchor.pkl'

with open(path, 'rb') as f:
    file = pickle.load(f)

print(file)
