def prec(rec):
  fields = list(rec.dtype.names)
  
  for i in range(rec[fields][0].size):
    s = ''
    for f in fields:
      s += str(rec[f][i]) 
    print s
