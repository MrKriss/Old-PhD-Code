import tables as tb

f = tb.openFile('C:\DataSets\kdd.h5','r')

mytab10 = f.root.RawData.KDD_10_Tab

mytabFull = f.root.RawData.KDD_Full_Tab

tabtypes = mytab10.coltypes

newFloatDic = {}

value = 'float32'
seq = []

for k, v  in tabtypes.iteritems():
    type(k)
    print type(v)
    #print k, v
    if v == value:
        seq.append(k)

newFloatDic.fromkeys(seq)
newFloatDic.update(tabtypes)

#f.close()

    
