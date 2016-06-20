from pylab import *
import numpy as np 
from filereading import *
import matplotlib as mpl 
from scipy.stats import gaussian_kde
def triple_compare():
	estems = np.loadtxt('est_ems.txt')
	testems = np.loadtxt('test_ems.txt')
	realems= np.loadtxt('real_ems.txt')
	color_scheme = np.loadtxt('color.txt')
	mcmap, norm = mpl.colors.from_levels_and_colors([-0.5,0.5,1.5,3], ['red','blue', 'green'])
	fig ,arr = subplots(1,2)
	faa = arr[0].scatter(realems[:,1],estems[:,1],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
	xs = np.linspace(-1,7,100)
	ys = xs
	hold(True)
	arr[0].plot(xs, ys,'--')
	arr[0].set_xlabel('Orig logEW')
	arr[0].set_ylabel('Reconstructed logEW')
	arr[0].set_title(r'My $H_{\beta}$')
	arr[0].set_xlim([-1,7])
	arr[0].set_ylim([-1,7])


	arr[1].scatter(realems[:,1],testems[:,1],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
	xs = np.linspace(-1,7,100)
	ys = xs
	hold(True)
	arr[1].plot(xs, ys,'--')
	arr[1].set_xlabel('Orig logEW')
	arr[1].set_ylabel('Reconstructed logEW')
	#arr[1].set_title(r'Beck $H_{\alpha}$')
	arr[1].set_title(r'My $H_{\beta}$')
	arr[1].set_xlim([-1,7])
	arr[1].set_ylim([-1,7])
	show()


def all_compare_11(nsample):
	
	namelist = ['$H_{\\alpha}$','$H_{\\beta}$','[OII]','[OIII] 4960','[NII]6550','[SII]6718','[OIII]5008','[NII]6585','[SII]6733']
	indslist = [0,1,5,7,9,2,6,8,3]
	testems = np.loadtxt('est_ems.txt')
	realems= np.loadtxt('real_ems.txt')
	real_sumarray = np.zeros((9,nsample))
	est_sumarray = np.zeros((9,nsample))
	errorsum = np.zeros(4)

	for i in np.arange(9):
		real_sumarray[i,:] = realems[:,indslist[i]]
		est_sumarray[i,:] = testems[:,indslist[i]]

	color_scheme = np.loadtxt('color.txt')
	mcmap, norm = mpl.colors.from_levels_and_colors([-0.5,0.5,1.5,3], ['red','blue', 'green'])
	xs = np.linspace(-3,7,100)
	ys = xs
	fig ,arr = subplots(3,3,sharex= True, sharey = True)
	subplots_adjust(hspace=0.0001)
	subplots_adjust(wspace=0.0001)

	for i in np.arange(9):
		inds = np.unravel_index(i,(3,3))
		j = inds[0]
		k = inds[1]
		h = arr[j][k].scatter(real_sumarray[i,:],est_sumarray[i,:],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
		hold(True)
		arr[j][k].plot(xs, ys,'r--')
		arr[j][k].set_xlim([-1.99,6])
		arr[j][k].set_ylim([-1.99,6])
		arr[j][k].legend([h],[namelist[i]])
		arr[j][k].set_xticks(np.arange(0, 6, 2.0))
		arr[j][k].set_yticks(np.arange(0, 6, 2.0))
	fig.text(0.5, 0.04, 'Orig logEW(in PCA basis)', ha='center', va='center')
	fig.text(0.06, 0.5, 'Reconstructed logEW', ha='center', va='center', rotation='vertical')
	show()
	fig ,arr = subplots(2,2,sharex= True, sharey = True)

	h = arr[0][0].scatter(real_sumarray[0,:],est_sumarray[0,:],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
	subplots_adjust(hspace=0.0001)
	subplots_adjust(wspace=0.0001)
	hold(True)
	arr[0][0].plot(xs, ys,'r--')
	arr[0][0].set_xlim([-1.99,6])
	arr[0][0].set_ylim([-1.99,6])
	arr[0][0].legend([h],[namelist[0]])
	arr[0][0].set_xticks(np.arange(0, 6, 2.0))
	arr[0][0].set_yticks(np.arange(0, 6, 2.0))
	fig.text(0.5, 0.04, 'Orig logEW(in PCA basis)', ha='center', va='center')
	fig.text(0.06, 0.5, 'Reconstructed logEW', ha='center', va='center', rotation='vertical')
	h = arr[0][1].scatter(real_sumarray[1,:],est_sumarray[1,:],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
	hold(True)
	arr[0][1].plot(xs, ys,'r--')
	arr[0][1].set_xlim([-1.99,6])
	arr[0][1].set_ylim([-1.99,6])
	arr[0][1].legend([h],[namelist[1]])
	arr[0][1].set_xticks(np.arange(0, 6, 2.0))
	arr[0][1].set_yticks(np.arange(0, 6, 2.0))
	fig.text(0.5, 0.04, 'Orig logEW(in PCA basis)', ha='center', va='center')
	fig.text(0.06, 0.5, 'Reconstructed logEW', ha='center', va='center', rotation='vertical')
	h = arr[1][0].scatter(real_sumarray[6,:],est_sumarray[6,:],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
	hold(True)
	arr[1][0].plot(xs, ys,'r--')
	arr[1][0].set_xlim([-1.99,6])
	arr[1][0].set_ylim([-1.99,6])
	arr[1][0].legend([h],[namelist[6]])
	arr[1][0].set_xticks(np.arange(0, 6, 2.0))
	arr[1][0].set_yticks(np.arange(0, 6, 2.0))
	fig.text(0.5, 0.04, 'Orig logEW(in PCA basis)', ha='center', va='center')
	fig.text(0.06, 0.5, 'Reconstructed logEW', ha='center', va='center', rotation='vertical')
	h = arr[1][1].scatter(real_sumarray[7,:],est_sumarray[7,:],c=color_scheme,cmap=mcmap,norm =norm,alpha=0.4,lw = 0, s=4)
	hold(True)
	arr[1][1].plot(xs, ys,'r--')
	arr[1][1].set_xlim([-1.99,6])
	arr[1][1].set_ylim([-1.99,6])
	arr[1][1].legend([h],[namelist[7]])
	arr[1][1].set_xticks(np.arange(0, 6, 2.0))
	arr[1][1].set_yticks(np.arange(0, 6, 2.0))
	fig.text(0.5, 0.04, 'Orig logEW(in PCA basis)', ha='center', va='center')
	fig.text(0.06, 0.5, 'Reconstructed logEW', ha='center', va='center', rotation='vertical')
	show()
	errorsum[0] = np.mean((real_sumarray[0,:]-est_sumarray[0,:])**2)
	errorsum[1] = np.mean((real_sumarray[1,:]-est_sumarray[1,:])**2)
	errorsum[2] = np.mean((real_sumarray[3,:]-est_sumarray[3,:])**2)
	errorsum[3] = np.mean((real_sumarray[7,:]-est_sumarray[7,:])**2)
	meanpred0 = np.mean((real_sumarray[0,:]-np.mean(real_sumarray[0,:]))**2)
	meanpred1 = np.mean((real_sumarray[1,:]-np.mean(real_sumarray[1,:]))**2)
	meanpred2 = np.mean((real_sumarray[3,:]-np.mean(real_sumarray[3,:]))**2)
	meanpred3 = np.mean((real_sumarray[7,:]-np.mean(real_sumarray[7,:]))**2)

	print "4 Emission Line MSE:",errorsum
	print "Compare Avg Prediction:", meanpred0,meanpred1,meanpred2,meanpred3

