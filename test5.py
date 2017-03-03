#coding:utf-8
from PIL import Image
import os
import math

class VectorCompare:
	#計算 矢量的大小：
	#
	def magnitude(self,concordance):
		total=0
		for word,count in concordance.iteritems():
			total=total+ count ** 2
		return math.sqrt(total)

	def relation(self,concordance1,concordance2):
		relevance=0;
		topvalue=0;
		for word,count in concordance1.iteritems():
			if concordance2.has_key(word):
				topvalue=topvalue+count*concordance2[word]
		return topvalue/ (self.magnitude(concordance1)*self.magnitude(concordance2))


#返回的是一個字典 key是count，（count是自加的，代辦像素的個數，value 是 像素數字 圖片的8bit一個像素，正好是0-255之間的元素）
def buildvector(im):
	d1={}
	#>>
	total=0
	#>>
	count=0
	for i in im.getdata():
		d1[count]=i
		#輸出的格式 ：85673 ****** (250)
		print count,"******",i
		count=count+1
	return d1

#構建一個訓練集
imageset=[]

temp_7=[]
temp_7.append( buildvector(Image.open("./iconset/7/4.gif")))
temp_7.append( buildvector(Image.open("./iconset/7/5b9e37490692eebbd4dcec3e437f3ec5.gif")))
temp_7.append( buildvector(Image.open("./iconset/7/13f34c3eac168090a288fbbb51cf8d95.gif")))
imageset.append({'7':temp_7})

temp_j=[]
temp_j.append( buildvector(Image.open("./iconset/j/8cd6e8039c58a9e5613e886ba7cdda5d.gif")))
temp_j.append( buildvector(Image.open("./iconset/j/d8150da674a65a6cfe194052a825383c.gif")))
imageset.append({'j':temp_j})

#構建一個測試集合
im_test=Image.open("j.gif")
bv_test=buildvector(im_test)


guess=[]
v=VectorCompare()

for Image in imageset:
	for x,y in Image.iteritems():
		g=v.relation(y[0],bv_test)
		guess.append((g,x))

guess.sort(reverse=True)
print ">>>"
print guess[0]

print ">>> 其他結果："
for f in guess:
	print f