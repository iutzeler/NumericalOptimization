import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
from IPython import display


def custom_3dplot( f, x1_min,x1_max,x2_min,x2_max,nb_points, v_min, v_max ):

	def f_no_vector(x1,x2):
		return f( np.array( [x1,x2] ) )

	x , y = np.meshgrid(np.linspace(x1_min,x1_max,nb_points),np.linspace(x2_min,x2_max,nb_points))
	z = f_no_vector(x,y)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(x, y, z,   cmap=cm.hot , vmin = v_min, vmax =  v_max)
	ax.set_zlim(v_min, v_max)
	plt.show()


def level_plot( f, x1_min,x1_max,x2_min,x2_max,nb_points, levels , title ):


	def f_no_vector(x1,x2):
		return f( np.array( [x1,x2] ) )

	x , y = np.meshgrid(np.linspace(x1_min,x1_max,nb_points),np.linspace(x2_min,x2_max,nb_points))
	z = f_no_vector(x,y)
	
	fig = plt.figure()
	graphe = plt.contour(x,y,z,levels)
	#plt.plot(3,1,'r*',markersize=15)
	plt.clabel(graphe,  inline=1, fontsize=10,fmt='%3.2f')
	plt.title(title)
	plt.show()


def level_points_plot( f , x_tab , x1_min,x1_max,x2_min,x2_max,nb_points, levels , title ):

	def f_no_vector(x1,x2):
		return f( np.array( [x1,x2] ) )

	x , y = np.meshgrid(np.linspace(x1_min,x1_max,nb_points),np.linspace(x2_min,x2_max,nb_points))
	z = f_no_vector(x,y)

	fig = plt.figure()
	graphe = plt.contour(x,y,z,levels)
	#plt.plot(3,1,'r*',markersize=15)
	plt.clabel(graphe,  inline=1, fontsize=10,fmt='%3.2f')
	plt.title(title)

	if x_tab.shape[0] > 40:
        	sub = int(x_tab.shape[0]/40.0)
        	x_tab = x_tab[::sub]
        
	delay = 2.0/x_tab.shape[0]
	for k in range(x_tab.shape[0]):
		plt.plot(x_tab[k,0],x_tab[k,1],'*b',markersize=10)
		#plt.annotate(k,(x_tab[k,0],x_tab[k,1]))
		plt.draw()	
		display.clear_output(wait=True)
		display.display(fig)
		time.sleep(delay)
	display.clear_output()
	plt.show()


def level_2points_plot( f , x_tab , x_tab2 , x1_min,x1_max,x2_min,x2_max,nb_points, levels , title ):


	def f_no_vector(x1,x2):
		return f( np.array( [x1,x2] ) )

	x , y = np.meshgrid(np.linspace(x1_min,x1_max,nb_points),np.linspace(x2_min,x2_max,nb_points))
	z = f_no_vector(x,y)

	fig = plt.figure()
	graphe = plt.contour(x,y,z,levels)
	#plt.plot(3,1,'r*',markersize=15)
	plt.clabel(graphe,  inline=1, fontsize=10,fmt='%3.2f')
	plt.xlim([x1_min,x1_max])
	plt.ylim([x2_min,x2_max])
	plt.title(title)

	if x_tab.shape[0] > 40:
        	sub = int(x_tab.shape[0]/40.0)
        	x_tab = x_tab[::sub]

	if x_tab2.shape[0] > 40:
        	sub = int(x_tab2.shape[0]/40.0)
        	x_tab2 = x_tab2[::sub]

	delay = 4.0/x_tab.shape[0]
	for k in range(x_tab.shape[0]):
		plt.plot(x_tab[k,0],x_tab[k,1],'*b',markersize=10)
		#plt.annotate(k,(x_tab[k,0],x_tab[k,1]))
		plt.draw()
		#plt.pause(delay)

	delay = 4.0/x_tab2.shape[0]
	for k in range(x_tab2.shape[0]):
		plt.plot(x_tab2[k,0],x_tab2[k,1],'dg',markersize=8)
		#plt.annotate(k,(x_tab2[k,0],x_tab2[k,1]))
		#plt.pause(delay)
		plt.draw()

	plt.show()
		


































