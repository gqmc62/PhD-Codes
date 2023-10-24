

import numpy as np
import matplotlib.pyplot as plt

color_map = plt.cm.get_cmap('Blues')
reversed_color_map = color_map.reversed()

IMT=np.zeros((16,18)) # response on WFS when a DM actuator is poked.

#WFS must have 3x3 up to inverse properly

#IMT[n]=[ x11,y11,x12,y12,x13,y13
#         x21,y21,x22,y22,x23,y23
#         x31,y31,x32,y32,x33,y33]

#first row of dm
IMT[0] =[ 1,-1, 0, 0, 0, 0,#first dm column
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]
IMT[1] =[-1,-1, 1,-1, 0, 0,#second dm column
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]
IMT[2] =[ 0, 0,-1,-1, 1,-1,#third dm column
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]
IMT[3] =[ 0, 0, 0, 0,-1,-1,#forth dm column
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]

#second row of dm
IMT[4] =[ 1, 1, 0, 0, 0, 0,
          1,-1, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]
IMT[5] =[-1, 1, 1, 1, 0, 0,
         -1,-1, 1,-1, 0, 0,
          0, 0, 0, 0, 0, 0]
IMT[6] =[ 0, 0,-1, 1, 1, 1,
          0, 0,-1,-1, 1,-1,
          0, 0, 0, 0, 0, 0]
IMT[7] =[ 0, 0, 0, 0,-1, 1,
          0, 0, 0, 0,-1,-1,
          0, 0, 0, 0, 0, 0]

#third row of dm
IMT[8] =[ 0, 0, 0, 0, 0, 0,
          1, 1, 0, 0, 0, 0,
          1,-1, 0, 0, 0, 0]
IMT[9] =[ 0, 0, 0, 0, 0, 0,
         -1, 1, 1, 1, 0, 0,
         -1,-1, 1,-1, 0, 0]
IMT[10]=[ 0, 0, 0, 0, 0, 0,
          0, 0,-1, 1, 1, 1,
          0, 0,-1,-1, 1,-1]
IMT[11]=[ 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0,-1, 1,
          0, 0, 0, 0,-1,-1]

#forth row of dm
IMT[12]=[ 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,
          1, 1, 0, 0, 0, 0]
IMT[13]=[ 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,
         -1, 1, 1, 1, 0, 0]
IMT[14]=[ 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,
          0, 0,-1, 1, 1, 1]
IMT[15]=[ 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0,-1, 1]

print('IMT=')
print(IMT)
print('\n')
plt.imshow(IMT)
plt.colorbar()
plt.title('IMT')
plt.xlabel('WFS')
plt.ylabel('DM')
plt.savefig('IMT')
plt.clf()

IM=np.transpose(IMT)

print('IM=')
print(IM)
print('\n')
plt.imshow(IM)
plt.colorbar()
plt.title('IM')
plt.xlabel('DM')
plt.ylabel('WFS')
plt.savefig('IM')
plt.clf()

IMm1=np.linalg.pinv(IM,rcond=0.7)

u,a,vt = np.linalg.svd(IM)
#pinv = u.transpose()*(1/a)*vt.transpose()
plt.plot(a)
plt.savefig('a')
plt.clf()


print('IMm1=')
print(IMm1)
print('\n')
plt.imshow(IMm1)
plt.colorbar()
plt.title('IMm1')
plt.xlabel('WFS')
plt.ylabel('DM')
plt.savefig('IMm1')
plt.clf()

DMin=np.transpose(np.asarray([0,0,0,0,
                              0,0.1,0,0,
                              0,0,0,0,
                              0,0,0,0]))
print('DMin=')
print(DMin)
print('\n')

WFSin=np.matmul(IM,DMin)
print('WFS response from test DM')
print(WFSin)
print('\n')

WFStest=WFSin
print('WFStest=')
print(WFStest)
print('\n')

plt.plot(WFStest[0]-1,WFStest[1]+1,marker='o')
plt.plot(WFStest[2]+0,WFStest[3]+1,marker='o')
plt.plot(WFStest[4]+1,WFStest[5]+1,marker='o')
plt.plot(WFStest[6]-1,WFStest[7]+0,marker='o')
plt.plot(WFStest[8]+0,WFStest[9]+0,marker='o')
plt.plot(WFStest[10]+1,WFStest[11]+0,marker='o')
plt.plot(WFStest[12]-1,WFStest[13]-1,marker='o')
plt.plot(WFStest[14]+0,WFStest[15]-1,marker='o')
plt.plot(WFStest[16]+1,WFStest[17]-1,marker='o')

plt.plot([-1,0,1,
          -1,0,1,
          -1,0,1],
         [1,1,1,
          0,0,0,
          -1,-1,-1],marker='x',color='k',ls='')
plt.plot([-1.5,1.5],[1.5,1.5],ls='-',color='k')
plt.plot([-1.5,1.5],[0.5,0.5],ls='-',color='k')
plt.plot([-1.5,1.5],[-0.5,-0.5],ls='-',color='k')
plt.plot([-1.5,1.5],[-1.5,-1.5],ls='-',color='k')

plt.plot([-1.5,-1.5],[-1.5,1.5],ls='-',color='k')
plt.plot([-0.5,-0.5],[-1.5,1.5],ls='-',color='k')
plt.plot([1.5,1.5],[-1.5,1.5],ls='-',color='k')
plt.plot([0.5,0.5],[-1.5,1.5],ls='-',color='k')

plt.axis('equal')
plt.title('WFSin')
plt.savefig('WFSin')
plt.clf()

DMans=np.matmul(IMm1,WFStest)
for i in np.arange(len(DMans)):
    DMans[i]=np.round(DMans[i],decimals=4)
print('DM response from test WFS')
print(DMans.reshape(4,4))
print('\n')

plt.imshow(DMin.reshape(4,4))
plt.colorbar()
plt.title('DMin')
plt.savefig('DMin')
plt.clf()

plt.imshow(DMans.reshape(4,4))
plt.colorbar()
plt.title('DMout')
plt.savefig('DMout')
plt.clf()

testinverse=np.matmul(np.linalg.inv(np.matmul(np.transpose(IM),IM)),np.transpose(IM))
plt.imshow(testinverse)
plt.savefig('testinverse')
plt.clf()

plt.imshow(IMm1-testinverse)


