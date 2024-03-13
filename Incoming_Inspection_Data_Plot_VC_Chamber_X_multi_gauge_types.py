#Python code for JP incoming inspection VCx Chamber data plotting
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
#import mplcursors
import numpy as np

root = tk.Tk()
root.withdraw()

# ファイル選択ダイアログを表示し、ファイルのパスを取得する
file_path = filedialog.askopenfilename(
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if file_path:
    # pandasを使ってCSVファイルをインポートする、1行目は飛ばして18行（つまり2～20行目まで）読み込み。
    dataini = pd.read_csv(file_path, skiprows = 1, nrows = 18, usecols=[0,1,2], header = None,names=['A', 'B', 'C'], skipinitialspace = True, encoding='SHIFT-JIS')
    #print(dataini.head(18))
else:
    print("No file selected.")

#列'A'から指定の定文字を含む行を抽出
rowdf = dataini[dataini['A'].str.contains('PN')]
#2列目(B列)値を抽出、文字列に変換
pn = rowdf.iloc[0,1]
pn = str(pn)
print('Part number:'+pn)
#検査対象ユニット数取得、整数値に変換
rowdf = dataini[dataini['A'].str.contains('Set')]
n_uut = int(rowdf['B'])
print('Number of UUT: %d' %n_uut)
#検査開始の日付を取得
rowdf = dataini[dataini['A'].str.contains('Start')]
date = rowdf.iloc[0,1]
print('Inspection date:'+date)
#元データのファイル名を取得
rowdf = dataini[dataini['A'].str.contains('File')]
sourcefilename = rowdf.iloc[0,1]
print('Source file:'+sourcefilename)
#ヘッダーデータによるオフセット行数を取得
headoffset = dataini[dataini['A'].str.contains('Point')].index.values[0] 
print('CSV Reading header: %d' %headoffset)

# ヘッダーのみを読み込む
df_header = pd.read_csv(file_path, skiprows= headoffset + 1, nrows=0)
print('[df.header]:')
print(df_header)

# `Humidity`列のインデックスを見つける
humidity_index = df_header.columns.get_loc("Humidity")
print('[Index of Humidity]:')
print(humidity_index)

# 必要な列のインデックス範囲を作成（`Point`が0番目の列）
cols = list(range(0, humidity_index + 1))

# 必要な列のみを読み込むために`usecols`パラメータを指定してCSVファイルを読み込む
header_df = pd.read_csv(file_path, skiprows = headoffset +1, usecols=cols ,nrows=1, dtype=str, encoding='SHIFT-JIS')
print(header_df)

#ヘッダーの1行目を表示
#print('header_df.head(1):')
#print(header_df.head(1))

#元データファイルからヘッドオフセットを除いたデータを読み込み
df = pd.read_csv(file_path, skiprows = headoffset+1,  usecols=cols , encoding='SHIFT-JIS')
#Convert datatype in 'Time' column to datetime for calculation
df['Time'] = pd.to_datetime(df['Time'], errors = 'coerce')
#データフレームの1列目が12以下の行を削除する（電圧データが0となっている部分を削除)
df = df[df['Point']>12]
#データプロットのサンプリングレートを計算（10ポイントに要する時間差から計算）
sec1 = df.iloc[0,1]
sec2 = df.iloc[10,1]
dtsec = sec2 - sec1
#print(dtsec)
secPerPoints = dtsec.seconds / 10
df['Time'] = df.index.values*secPerPoints

print('df:')
print(df.head(3))

#Make array with numbers [3,5,7...41.43]
columns_to_transform = range(3,2 + n_uut*2,2)
print('Colums to convert voltage to pressure')
print(columns_to_transform)

#ヘッダーの後ろから2番目の列のインデックス数(size - 1)を取得
#col_temperature = header_df.size-2
#print('col temp:%d' %col_temperature)
#print(df.iloc[0, col_temperature])

#Default spec for BPG400 and 402

if pn == '353-500' or pn == '353-570' or pn == '353-576' or pn == '353-577':# BPG400, BPG402 type
    df.iloc[:,columns_to_transform] =df.iloc[:,columns_to_transform].apply(lambda x: 10**((x-7.75)/0.75)) 
    print('df(after v to p conversion):')
    print(df.head(3))
    spec_group = 2
    if pn == '353-550' or pn == '353-551': #BCG450 type
        spec_group = 1

if pn == '353-497' or pn == '355-478' or pn == '355-495' or pn == '355-574' or pn == '3PC1-001-3000' or pn == '3PC1-001-3000T':# PCG type
    df.iloc[:,columns_to_transform] =df.iloc[:,columns_to_transform].apply(lambda x: 10**(0.7777778*(x-6.1428571))) #PCG 0.61V...10.23V type
    print('df(after v to p conversion):')
    print(df.head(3))
    spec_group = 3
    if pn == '350-140': #Normal Pirani type
        spec_group = 4

if spec_group >= 1 or spec_group <= 4:  #This part will be modified for the future use of CDG references
    ref_values = df.loc[:,'BPG400'].apply(lambda x: 10**((x-7.75)/0.75)) #.rolling(7).mean().round(1).fillna(0)
    ref1name = 'BPG400'
    for x in ref_values:
        A = np.array(ref_values)
        refP = A
        refN = A
    #row, col = A.shape
    print(A)


if spec_group ==1 : #BCG450 type
    refP = np.where(refP>1050,0,refP) #np.where(condition, true case, false case)
    refP = np.where( (refP<=1050) & (refP>=950), refP + refP*0.025, refP )
    refP = np.where( (refP<950) & (refP>=50), refP + refP*0.05, refP )
    refP = np.where( (refP<50) & (refP>=1e-8), refP + refP*0.15, refP )
    refP = np.where(refP<1e-8,0,refP)
    refP = np.where(refP==0,None,refP)

    refN = np.where(refN>1050,0,refN) #np.where(condition, true case, false case)
    refN = np.where( (refN<=1050) & (refN>=950), refN - refN*0.025, refN )
    refN = np.where( (refN<950) & (refN>=50), refN - refN*0.05, refN )
    refN = np.where( (refN<50) & (refN>=1e-8), refN - refN*0.15, refN )
    refN = np.where(refN<1e-8,0,refN)
    refN = np.where(refN==0,None,refN)
    

if spec_group ==2 : #BPG402 type
    refP = np.where(refP>0.01,0,refP) #np.where(condition, true case, false case)
    refP = np.where( (refP<=0.01) & (refP>=1e-8), refP + refP*0.15, refP )
    refP = np.where(refP<1e-8,0,refP)
    refP = np.where(refP==0,None,refP)

    refN = np.where(refN>0.01,0,refN) #np.where(condition, true case, false case)
    refN = np.where( (refN<=0.01) & (refN>=1e-8), refN - refN*0.15, refN )
    refN = np.where(refN<1e-8,0,refN)
    refN = np.where(refN==0,None,refN)
    #SpecN = np.array( (A<0.01) & (A>=1e-8), A-A*0.15,SpecP )
    print(refP)

if spec_group ==3 : #PCG type
    refP = np.where(refP>1050,0,refP) #np.where(condition, true case, false case)
    refP = np.where( (refP<=1050) & (refP>=950), refP + refP*0.025, refP )
    refP = np.where( (refP<950) & (refP>=100), refP + refP*0.05, refP )
    refP = np.where( (refP<100) & (refP>=1e-3), refP + refP*0.15, refP )
    refP = np.where( (refP<1e-3) & (refP>=5e-4), refP + refP*0.5, refP )
    refP = np.where(refP<5e-4,0,refP)
    refP = np.where(refP==0,None,refP)

    refN = np.where(refN>1050,0,refN) #np.where(condition, true case, false case)
    refN = np.where( (refN<=1050) & (refN>=950), refN - refN*0.025, refN )
    refN = np.where( (refN<950) & (refN>=100), refN - refN*0.05, refN )
    refN = np.where( (refN<100) & (refN>=1e-3), refN - refN*0.15, refN )
    refN = np.where( (refN<1e-3) & (refN>=5e-4), refN - refN*0.5, refN )
    refN = np.where(refN<5e-4,0,refN)
    refN = np.where(refN==0,None,refN)

if spec_group ==4 : #Pirani type
    refP = np.where(refP>1000,0,refP) #np.where(condition, true case, false case)
    refP = np.where( (refP<=1000) & (refP>=100), refP + refP*0.5, refP )
    refP = np.where( (refP<100) & (refP>=1e-3), refP + refP*0.15, refP )
    refP = np.where( (refP<1e-3) & (refP>=5e-4), refP + refP*0.5, refP )
    refP = np.where(refP<5e-4,0,refP)
    refP = np.where(refP==0,None,refP)

    refN = np.where(refN>1000,0,refN) #np.where(condition, true case, false case)
    refN = np.where( (refN<=1000) & (refN>=100), refN - refN*0.5, refN )
    refN = np.where( (refN<100) & (refN>=1e-3), refN - refN*0.15, refN )
    refN = np.where( (refN<1e-3) & (refN>=5e-4), refN - refN*0.5, refN )
    refN = np.where(refN<5e-4,0,refN)
    refN = np.where(refN==0,None,refN)

column_names = list(df.columns)

# プロットを作成する
fig, ax = plt.subplots()
ax2 = ax.twinx()
time = df.loc[:,'Time']
temp = df.loc[:,'Temperature']
temp = temp.replace(0,None)
humid = df.loc[:,'Humidity']
humid = humid.replace(0,None)
half_humid = humid * 0.5

lines = []
for i, item in enumerate(columns_to_transform):
    line_item, = ax.plot(time, df.iloc[:,item], label = column_names[item])
    lines.append(line_item)

line_item, = ax.plot(time, ref_values, label = ref1name)
lines.append(line_item)
line_item, = ax.plot(time, refP, "--",dashes=[1,100,1,100] ,c='m', linewidth=2, label = 'ref_acc%+')
lines.append(line_item)
line_item, = ax.plot(time, refN, "--",dashes=[1,100,1,100] ,c='m', linewidth=2, label = 'ref_acc%-')
lines.append(line_item)


ax2.set_ylim(0,50)
ax2.plot(time, temp, ":", label = 'T', c="tab:orange")
ax2.plot(time, half_humid, ":", label = 'H', c="blue")
ax2.set_ylabel('Temperature(°C) / Humidity/2(RH%)')

legendlocation = 0
  #legend loc options ->>>: 'best' 0, 'upper right' 1, 'upper left' 2, 'lower left' 3, 'lower right' 4, 'right' 5, 
  #'center left'	6, 'center right'	7, 'lower center'	8, 'upper center'	9,'center'	10

ymax = 1500
ymin = 1E-7
ax.set(ylim=(ymin, ymax))
ax.grid()
ax.grid(which = "minor")
ax.minorticks_on()
ax.set_yscale('log')
ax.set_ylabel('Pressure(mbar)')
ax.set_xlabel('Time(s)')

h,l=ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg1 = ax.legend(h, l , loc='best', fancybox=True, shadow=False)
#Set transparency of legend box
leg1.get_frame().set_alpha(0.5)
leg2 = ax2.legend(loc="lower right")
leg2.get_frame().set_alpha(0.5)
#ax.add_artist(leg1)


ax.set_zorder(2) #Put the primary axis to upper layer
ax2.set_zorder(1)
ax.patch.set_alpha(0) #(patch) is set to 0 transparency, so the background become completely transparent

lined = dict()
for legline, origline in zip(leg1.get_lines(), lines):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = origline



def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', onpick)

atext = ''

def onclick(event):
    #plt.sca(ax) # Set current axis to ax1 before checking event.inaxes
    global ax
    if event.xdata == None or event.ydata == None:
        return
    else:
        print('{} click: button={}, x={}, y={}, xdata={}, ydata={}'.format(
        'double' if event.dblclick else 'single', event.button,
         event.x, event.y, event.xdata, event.ydata,
        ))
        if event.dblclick:
            x=event.xdata
            y=event.ydata
            text= ' '
            ax.annotate(text, xy=(x,y), xytext=(x,y), fontsize=15,color='r', arrowprops=dict(facecolor='red',
            width=1, headwidth=10, headlength=5, edgecolor='red',))
            # global cid_onclick
            # fig.canvas.mpl_disconnect(cid_onclick)
            # cid_onclick = None
            fig.canvas.draw()
        if event.button ==3:
            x=event.xdata
            y=event.ydata
            text= atext
            global ann
            ann = ax.annotate(text, xy=(x,y), xytext=(x,y-y*0.1), fontsize=15,color='r', arrowprops=dict(facecolor='red',
            width=1, headwidth=10, headlength=5, edgecolor='red',))
            #global cid_onclick
            #fig.canvas.mpl_disconnect(cid_onclick)
            #cid_onclick = None
            fig.canvas.draw()



# Below are codes to activate / deactivate onclick function (系列ラインON OFFの障害になるケースが有ったため追加したコード)
cid_onclick = None

cid_onclick = fig.canvas.mpl_connect('button_press_event', onclick)

def connect(event):
    global cid_onclick
    cid_onclick = fig.canvas.mpl_connect('button_press_event', onclick)

def disconnect(event):
    global cid_onclick
    fig.canvas.mpl_disconnect(cid_onclick)
    cid_onclick = None

def toggle_legends(event):
    global cid_onclick
    # toggle visibility of all lines
    for l in lined:
        origline = lined[l]
        #print(origline)
        vis = not origline.get_visible()
        lined[l].set_visible(vis)
        if vis:
            l.set_alpha(1.0)
        else:
            l.set_alpha(0.2)
    # update plot canvas
    fig.canvas.draw()

def ann_removeal(event):
    ann.remove()
    fig.canvas.draw()

def delete_text(event):
    text_box.set_val('')

# axset = fig.add_axes([0.3, 0.02, 0.07, 0.05])
# bset = Button(axset, 'onclick')
# bset.on_clicked(connect)

# axback = fig.add_axes([0.4, 0.02, 0.07, 0.05])
# bback = Button(axback, 'release')
# bback.on_clicked(disconnect)

axdeltext = fig.add_axes([0.35, 0.02, 0.05, 0.05])
deltext = Button(axdeltext, 'del txt')
deltext.on_clicked(delete_text)

axannrem = fig.add_axes([0.4, 0.02, 0.05, 0.05])
bannrem = Button(axannrem, 'del ann')
bannrem.on_clicked(ann_removeal)

axlegtoggle = axset = fig.add_axes([0.8, 0.02, 0.07, 0.05])
blegtoggle = Button(axlegtoggle, 'toggle legends')
blegtoggle.on_clicked(toggle_legends)


def annotation_text(text):
    global atext
    atext = text
    #print(atext)

axbox = fig.add_axes([0.2, 0.02, 0.15, 0.05])
text_box = TextBox(axbox, "right click in plot", textalignment="left")
text_box.on_text_change(annotation_text)


fig = plt.gcf()    # Get current figure
fig.set_size_inches(16,8)   # Set size in inch
plt.suptitle(pn + ' Incoming inspection ' + date)
plt.show()

