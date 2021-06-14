import pandas as pd 

class FeatureReport:
    def __init__(self, series):
        self.name = series.name
        self.count = series.count()
        self.pct_na = series.isna().sum() / self.count
        self.card = series.nunique()
        
class NumericalFeatureReport(FeatureReport):
    def __init__(self, series):
        FeatureReport.__init__(self, series)
        self.min = series.min()
        self.q1 = series.quantile(.25)
        self.mean = series.mean()
        self.median = series.quantile(.5)
        self.q3 = series.quantile(.75)
        self.max = series.max()
        self.std = series.std()
        
class CategoricalFeatureReport(FeatureReport):
    def __init__(self, series):
        FeatureReport.__init__(self, series)
        self.is_id = self.card == self.count
        if self.is_id:
            self.mode = None
            self.mode_freq = None
            self.mode_pct = None
            self.mode2 = None
            self.mode2_freq = None
            self.mode2_pct = None
        else:
            value_counts = series.value_counts()
            self.mode = value_counts.index[0]
            self.mode_freq = value_counts[self.mode]
            self.mode_pct = self.mode_freq / self.count
            self.mode2 = value_counts.index[1]
            self.mode2_freq = value_counts[self.mode2]
            self.mode2_pct = self.mode2_freq / self.count
            
class DataQualityReport:
    def __init__(self, df, feature_type='numerical'):
        fr_dfs = []
        if feature_type == 'numerical':
            for c in df.columns:
                fr = NumericalFeatureReport(df[c])
                fr_df = pd.DataFrame(fr.__dict__, index=[fr.name]).drop(columns=['name'])
                fr_dfs.append(fr_df)
        elif feature_type == 'categorical':
            for c in df.columns:
                fr = CategoricalFeatureReport(df[c])
                fr_df = pd.DataFrame(fr.__dict__, index=[fr.name]).drop(columns=['name', 'is_id'])
                fr_dfs.append(fr_df)
        self.summary = pd.concat(fr_dfs)