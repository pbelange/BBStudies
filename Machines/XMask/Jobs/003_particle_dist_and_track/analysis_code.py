


# Computing sigma
sig_x_idx = np.abs(tracked.df_sig.x_sig-1).argmin()
sig_y_idx = np.abs(tracked.df_sig.y_sig-1).argmin()
sig_x     = tracked.df.loc[sig_x_idx,'x']
sig_y     = tracked.df.loc[sig_y_idx,'y']


coll_x=6*sig_x
coll_y=6*sig_y
coll_s=20*sig_x

def lost_condition(x_min,y_min,x_max,y_max,skew_angle = 127.5):
    # y_fun_skew = lambda _x: np.tan(np.deg2rad(skew_angle))*_x + coll_s/np.cos(np.deg2rad(180-skew_angle))
    # np.abs(y)>y_fun_skew(np.abs(x)))
    return (np.abs(x_min)>coll_x)|(np.abs(y_min)>coll_y)|(np.abs(x_max)>coll_x)|(np.abs(y_max)>coll_y)


# df = tracked.df[['turn','particle','x_min','y_min']]
_lost  = lost_condition(tracked.data.x_min,tracked.data.y_min,tracked.data.x_max,tracked.data.y_max)
idx_lost     = tracked.data.index[_lost]
idx_survived = tracked.data.index[~_lost]


# New columns
try:
    tracked.data.insert(0,'beyond_coll',False)
    tracked.data.insert(0,'lost',False)
except:
    tracked.data.loc[:,'beyond_coll'] = False
    tracked.data.loc[:,'lost'] = False



tracked.data.loc[idx_lost,'beyond_coll'] = True
tracked.data.loc[:,'lost'] = tracked.data.groupby('particle').beyond_coll.cumsum().astype(bool)

Intensity = tracked.data[~tracked.data.lost].groupby('start_at_turn').count().lost
