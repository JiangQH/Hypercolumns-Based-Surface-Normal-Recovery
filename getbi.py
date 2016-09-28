from math import floor, ceil
def getbilinear(inputs, W_=224):
	widths = (224,112,56,28,14,1)
	h = inputs / W_;
	w = inputs % W_; 
	for width in widths:
		scale = W_ / width
		pad = (scale-1.0)/2.0
		tempw = (w - pad) / scale
		temph = (h - pad) / scale
		fw = floor(tempw)
		fh = floor(temph)
		cw = ceil(tempw)
		ch = ceil(temph)
                if fw < 0:
                    fw = 0
                if fh < 0:
                    fh = 0
                if cw < 0:
                    cw = 0
                if ch < 0:
                    ch = 0
                if cw > width:
                    cw = width
                if ch > width:
                    ch = width
		if (fw==cw) and (fh==ch):
			print "for bottom width ",  w, fh*width+fw, 1
		elif (fh==ch):
			print "for bottom width " , w, fh*width+fw, (1-tempw+fw), fh*width+fw+1, tempw-fw
		elif (fw==cw):
			print "for bottom width ",  w, fh*width+fw, (1-temph+fh), fh*width+fw+width, temph-fh
		else:
			print "for bottom width ",  w, fh*width+fw, (1-temph+fh)*(1-tempw+fw), fh*width+fw+1, (1-temph+fh)*(tempw-fw), fh*width+fw+width, (temph-fh)*(1-tempw+fw), fh*width+fw+width+1, (temph-fh)*(tempw-fw)
