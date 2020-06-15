

class Identity():
    def __init__(self, df, mode='train'):
        assert mode == 'train' or mode == 'valid' or mode == 'test'
        self.mode = mode
        self.df = df

    def get_feature(self):
        return self.df.copy(deep=True)
