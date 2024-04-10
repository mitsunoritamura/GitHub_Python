import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
import os
import datetime

# Create a root window and hide it
root = tk.Tk()
root.withdraw()


# Open a file dialog and get the path of the selected file
file_path = filedialog.askopenfilename()
# To get the file name, we can use os.path.basename
file_name = os.path.basename(file_path)

# Load the data from column0 and column1
dfini = pd.read_csv(file_path, usecols=[0,1], encoding='SHIFT-JIS').head(100)
#dfini = pd.read_csv(file_path, error_bad_lines=False, encoding='SHIFT-JIS').head(100)
# Delete first column

filtered_rows = dfini[dfini['#BeginHeader'].str.contains('#EndHeader', na=False)]#エラーを回避するためには、naパラメータを使用して、str.contains()がNaNを返す代わりに特定のBoolean値を返すように指定
index_values = filtered_rows.index.tolist()
headoffset = index_values
print(headoffset)

filtered_rows = dfini[dfini['#BeginHeader'].str.contains('No. of Data', na=False)]
index_values = filtered_rows.index.tolist()
Nofdataraw = index_values
print(Nofdataraw)

Nofdata = pd.to_numeric(dfini.iloc[Nofdataraw,1]).iloc[0] #Nofdatacol becomes 'str' type, need to convert to number
print('Nofdata=')
print(Nofdata)
print(type(Nofdata))


dfini2 = pd.read_csv(file_path, skiprows=34, nrows = 5,  encoding='SHIFT-JIS')
print(dfini2)

#Wavelogger内で付けられたチャンネル名を取得
waveform_name_list = dfini2[dfini2.apply(lambda row: row.isin(['Waveform Name']).any(), axis=1)]
print(waveform_name_list)

# 'CDG1000'の値がある列のインデックスを取得
CDG1000_index = waveform_name_list.columns[(waveform_name_list == 'CDG1000').iloc[0]].tolist()
print(CDG1000_index)
# 'CDG1000'の値がある列のインデックス番号を取得
CDG1000_index_number = [waveform_name_list.columns.get_loc(col) for col in CDG1000_index]
print(CDG1000_index_number)

# 'CDG10'の値がある列のインデックスを取得
CDG10_index = waveform_name_list.columns[(waveform_name_list == 'CDG10').iloc[0]].tolist()
print(CDG10_index)
# 'CDG10'の値がある列のインデックス番号を取得
CDG10_index_number = [waveform_name_list.columns.get_loc(col) for col in CDG10_index]
print(CDG10_index_number)

# 'CDG0.1'の値がある列のインデックスを取得
CDG0_1_index = waveform_name_list.columns[(waveform_name_list == 'CDG0.1').iloc[0]].tolist()
print(CDG0_1_index)
# 'CDG0.1'の値がある列のインデックス番号を取得
CDG0_1_index_number = [waveform_name_list.columns.get_loc(col) for col in CDG0_1_index]
print(CDG0_1_index_number)


#データ部分を読み込む
# 0-indexedなので、2-4行目は1-3、10-20行目は9-19として指定
#列名を'Waveform Name行から取得
skip_rows = list(range(0, 35)) + list(range(36, 62))
df = pd.read_csv(file_path, skiprows=skip_rows, nrows=Nofdata, encoding='SHIFT-JIS')#, engine='python')
print(df)
#print(df.head(100))


CDG_lowest_press = 1500

if not CDG1000_index:
    print("CDG1000_index is an empty list.")
    P_CDG1000_2decades = np.array([]) #Making a List with size = 0 to avoid Name Error for not existing valuable
else:
    print("CDG1000_index is not empty.")
    df.iloc[:,CDG1000_index_number] = df.iloc[:, CDG1000_index_number].apply(lambda x: (x*1000/10))
    P_CDG1000 = df.iloc[:,CDG1000_index_number]
    P_CDG1000_2decades = np.array(P_CDG1000)
    P_CDG1000_2decades = np.where(P_CDG1000_2decades<1,-1,P_CDG1000_2decades)
    CDG_lowest_press = 10
    P_CDG1000_2decades = np.where(P_CDG1000_2decades>1020,0,P_CDG1000_2decades) #np.where(condition, true case, false case)

    #print(*P_CDG1000_2decades)

if not CDG10_index:
    print("CDG10_index is an empty list.")
    P_CDG10_2decades = np.array([]) #Making a List with size = 0 to avoid Name Error for not existing valuable
else:
    print("CDG10_index is not empty.")
    #df.iloc[:,CDG10_index_number] = df.iloc[:, CDG10_index_number].apply(lambda x: (x*10/10))  #Don't have to convert for 10T gauge
    P_CDG10 = df.iloc[:,CDG10_index_number]
    P_CDG10_2decades = np.array(P_CDG10)
    P_CDG10_2decades = np.where(P_CDG10_2decades<0.01,-1,P_CDG10_2decades)
    P_CDG10_2decades = np.where(P_CDG10_2decades>10.02,0,P_CDG10_2decades) #np.where(condition, true case, false case)
    CDG_lowest_press = 0.1

    #print(*P_CDG10_2decades)

if not CDG0_1_index:
    print("CDG0.1_index is an empty list.")
    P_CDG0_1_2decades = np.array([]) #Making a List with size = 0 to avoid Name Error for not existing valuable
else:
    print("CDG0.1_index is not empty.")
    low_cut_value = 0.00010
    df.iloc[:,CDG0_1_index_number] = df.iloc[:, CDG0_1_index_number].apply(lambda x: (x*0.01))
    P_CDG0_1 = df.iloc[:,CDG0_1_index_number]
    P_CDG0_1_2decades = np.array(P_CDG0_1)
    P_CDG0_1_2decades = np.where(P_CDG0_1_2decades<low_cut_value,-1,P_CDG0_1_2decades)
    P_CDG0_1_2decades = np.where(P_CDG0_1_2decades>0.102,0,P_CDG0_1_2decades) #np.where(condition, true case, false case)
    CDG_lowest_press = low_cut_value

    #print(*P_CDG0_1_2decades)

if CDG0_1_index:
    P_CDG_combined = np.array(P_CDG0_1_2decades)
    print("CDG0.1 pressure combined.")
else:
    if CDG10_index:
        P_CDG_combined = np.array(P_CDG10_2decades)
        print("CDG10 pressure combined.")
    else:
        if CDG1000_index:
            P_CDG_combined = np.array(P_CDG1000_2decades)
            print("CDG1000 pressure combined.")
        else:
            print("No CDG pressure combined.")

<<<<<<< HEAD
if P_CDG_combined.size > 0:
        if P_CDG10_2decades.size > 0:
            P_CDG_combined = np.where(P_CDG_combined==0,P_CDG10_2decades,P_CDG_combined)
=======

if not P_CDG1000_2decades:
    print("CDG1000_2decade is an empty list.")
else:
    if P_CDG_combined.size > 0:
        if P_CDG10_2decades.size > 0:
        P_CDG_combined = np.where(P_CDG_combined==0,P_CDG10_2decades,P_CDG_combined)
>>>>>>> 38584bd7cde5ee09fa310941e9e7838150bf50ca
        if P_CDG1000_2decades.size > 0:
            P_CDG_combined = np.where(P_CDG_combined==0,P_CDG1000_2decades,P_CDG_combined)
            P_CDG_combined = np.where(P_CDG_combined==-1,CDG_lowest_press,P_CDG_combined)
    
#print(*P_CDG_combined)


#Convert argument to a numeric type.
df['Unnamed: 1'] = pd.to_numeric(df['Unnamed: 1'], errors = 'coerce')
#print(df.iloc[0,0])
#print(type(df.iloc[0,0]))
tdstr1 = df.iloc[0,0]
td1 = datetime.datetime.strptime(tdstr1,'%m/%d/%Y %H:%M:%S')
#print(td1)
#print(type(td1))
tdstr2 = df.iloc[1,0]
td2 = datetime.datetime.strptime(tdstr2,'%m/%d/%Y %H:%M:%S')
dt = td2 - td1
print(dt)
print(type(dt))
dtsecN = dt.seconds #this includes hours and minutes but not days
dtmsec = (df.iloc[1,1] - df.iloc[0,1] ) / 1000000
df['Unnamed: 1'] = df.index.values * (dtsecN + dtmsec)
df.rename(columns={'Unnamed: 1':'time'})

target_columns = [col for col in df.columns if col.startswith('BPG')]
for col in target_columns:
    df[col] = pd.to_numeric(df[col], errors = 'coerce')
    df[col] = (10**((df[col]-7.75)/0.75))

target_columns = [col for col in df.columns if col.startswith('BCG') or col.startswith('ULVE')]
for col in target_columns:
    df[col] = pd.to_numeric(df[col], errors = 'coerce')
    df[col] = (10**((df[col]-7.75)/0.75))

target_columns = [col for col in df.columns if col.startswith('CDG')]
for col in target_columns:
    df= df.drop(columns=[col])

df['ref(CDG0.1-1000T)'] = P_CDG_combined*1.33322

df.head()

#Use gas type coefficience depending on pressure ranges
usePeffArgon = False
usePeffHelium = False
usePeffPBMixHe = False
addPlotTitle = ''
useref = False


# Define color scheme for each series
color_scheme = {
    "ULVETANNA": plt.cm.Blues,
    "BCG": plt.cm.Reds,
    "BPG": plt.cm.Greens,
    "Channel": plt.cm.Purples
}


# Create a figure and an axis
fig, ax = plt.subplots(figsize=(10, 6))

#系列のkeyと数を取得、カラーマップリストにインデックス数分を挿入
cmapnum = []
for i, key in enumerate(color_scheme.keys()):
    cmapnum.insert(i,i)
#2列目以降の列数を取得
for i, column in enumerate(df.columns[1:]):
    maxindex = i

#ガス種依存性分を補正する場合の係数とグラフタイトルに追加する文字を設定
if usePeffArgon == True:
    coefPir = 1.7
    coefBA = 0.8
    addPlotTitle = ' -Peff = Ar'
elif usePeffHelium == True:
    coefPir = 1.2 #0.8 #0.8 is from our operation manual
    coefBA = 5.9
    addPlotTitle = ' -Peff = He coefPir 1.2'
else :
    coefPir = 1.0
    coefBA = 1.0

if coefPir != 1.0:
    for i, column in enumerate(df.columns[1:]):
        df.loc[(df['Ref. Pressure'] >= 0.02) & (df['Ref. Pressure'] <= 1), column] *= coefPir

if coefBA != 1.0:
    for i, column in enumerate(df.columns[1:]):
        df.loc[(df['Ref. Pressure'] < 0.0055), column] *= coefBA     

if usePeffPBMixHe == True:
          for i, column in enumerate(df.columns[1:]):
            df.loc[(df['Ref. Pressure'] >= 0.0055) & (df['Ref. Pressure'] < 0.02), column] *= df['Ref. Pressure'] * -285 + 7.21


if useref == True:
    refP = np.array(P_CDG_combined)
    refP = np.where(refP>1050,0,refP) #np.where(condition, true case, false case)
    refP = np.where( (refP<=1050) & (refP>=950), refP + refP*0.025, refP )
    refP = np.where( (refP<950) & (refP>=50), refP + refP*0.05, refP )
    refP = np.where( (refP<50) & (refP>=1e-8), refP + refP*0.15, refP )
    refP = np.where(refP<1e-8,0,refP)
    refP = np.where(refP==0,None,refP)

    refN = df["Ref. Pressure"]
    refN = np.where(refN>1050,0,refN) #np.where(condition, true case, false case)
    refN = np.where( (refN<=1050) & (refN>=950), refN - refN*0.025, refN )
    refN = np.where( (refN<950) & (refN>=50), refN - refN*0.05, refN )
    refN = np.where( (refN<50) & (refN>=1e-8), refN - refN*0.15, refN )
    refN = np.where(refN<1e-8,0,refN)
    refN = np.where(refN==0,None,refN)

    df['spec P'] = refP
    df['spec N'] = refN


for i, column in enumerate(df):
    color = "black"
    marker = 'o'
    s = 3.0

for i, column in enumerate(df):

    for j, key in enumerate(color_scheme.keys()):
        if column.startswith(key):
            cmap = color_scheme[key]
            # Use a different shade for each series within the same category
            color = cmap((cmapnum[j] % maxindex + 1) / maxindex)
            cmapnum[j] = cmapnum[j] +1 
            marker = 'o'
            s = 3.0
            break
            
    if column == 'ref(CDG0.1-1000T)' :
        color = 'Magenta'
        marker = 'o'
        s = 3.0 

    if i>1:
        x_values = df["Unnamed: 1"]
        y_values = df[column]
        ax.plot(x_values, y_values, color=color, marker=marker, markersize=s, label=column)
        #ax.scatter(x_values, y_values, color=color, marker=marker, label=column, s=s)


# Set the scale of both axes to logarithmic
#ax.set_xscale('log')
ax.set_yscale('log')


# Add minor ticks on the log scale
#ax.xaxis.set_minor_locator(LogLocator(numticks=13, subs=(.1,.2,.3,.4,.5,.6,.7,.8,.9)))
ax.yaxis.set_minor_locator(LogLocator(numticks=13, subs=(.1,.2,.3,.4,.5,.6,.7,.8,.9)))

# Draw minor grid lines
ax.grid(which='minor', linestyle='-', linewidth='0.5', color='gray')
ax.xaxis.grid(True, which='major')  # X軸の主目盛りに対してグリッド線を表示
ax.yaxis.grid(True, which='major')  # Y軸の補助目盛りに対してグリッド線を表示

# Increase the size of minor ticks
ax.tick_params(which='minor', width=1)

# Set labels and title
ax.set_xlabel('Time (sec)', fontsize=12)
ax.set_ylabel('Value (mbar)', fontsize=12)
ax.set_title(file_name + addPlotTitle, fontsize=15)

# Add a legend
ax.legend()

# Show the plot
plt.show()

