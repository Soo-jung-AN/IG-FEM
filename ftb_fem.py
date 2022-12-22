import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as triang
from scipy.spatial import Delaunay
import numpy.linalg as lina
from Assembly import M_assembly, A_assembly, R_assembly
from preprocessing import reshape, Get_shf_coef, Get_gp_cood
from scipy import sparse
from pypardiso import spsolve
np.set_printoptions(precision=10, threshold=20000000, linewidth=20000000)
############################################################################################################################################################
#TODO control
init, second, third, forth, last = 5932, 30283, 42336, 54402, 66494
undeform = init
deform = second
undeformed_cood = np.loadtxt("cood0_{}.txt".format(undeform))
deformed_cood = np.loadtxt("cood0_{}.txt".format(deform))
spin = np.loadtxt("angular_vel0_{}.txt".format(42336))#28204
rad = np.loadtxt("radius_30699.txt")
############################################################################################################################################################
p_num = len(undeformed_cood)
disp = deformed_cood - undeformed_cood
u_disp = disp[:,0]; v_disp = disp[:,1]
############################################################################################################################################################
# TODO triangulation
triang = triang.Triangulation(undeformed_cood[:,0],undeformed_cood[:,1])
#tt = triang.edges
#tt = reshape(undeformed_cood, tt, 115)
tri = Delaunay(undeformed_cood)
ele_id = tri.simplices
ele_id = reshape(undeformed_cood, ele_id, 115)
TT_E = len(ele_id); E_area = np.zeros(TT_E)
# plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id)
# plt.gca().set_aspect(1)
# plt.show()
#################################################################################
# TODO Solving method
import time
solving_time = time.time()
SC_mat_e = np.zeros((TT_E,3,3), dtype=np.float64)
Get_shf_coef(SC_mat_e, ele_id, undeformed_cood)
PQ_detJ_e = np.zeros((TT_E,3,3), dtype=np.float64)
Get_gp_cood(PQ_detJ_e, ele_id, undeformed_cood)

M_RC = np.zeros((2,36 * TT_E), dtype=np.int64); M_data = np.zeros(36 * TT_E, dtype=np.float64)
M_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, M_RC, M_data, p_num)
M_CSR = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num * 4, p_num * 4))

A_RC = np.zeros((2,36 * TT_E), dtype=np.int64); A_data = np.zeros(36 * TT_E, dtype=np.float64)
A_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, A_RC, A_data, p_num)
A_CSR = sparse.csr_matrix((A_data, (A_RC[0], A_RC[1])), shape=(p_num * 4, p_num * 4))

U = np.hstack((u_disp,v_disp))
U = np.hstack((U,U))

AU = A_CSR * U

R_vec = np.zeros(p_num * 4, dtype=np.float64)
R_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, R_vec, p_num)

AU_R = AU + R_vec

F = spsolve(M_CSR, AU_R)

E = 0.5 * (np.transpose(F) @ F - R_vec)
print("elasped time:",time.time()-solving_time)

solved_F11 = F[:p_num]
solved_F22 = F[p_num:2*p_num]
solved_F12 = F[2*p_num:3*p_num]
solved_F21 = F[3*p_num:]

aUaX = solved_F11 - 1
aUaY = solved_F12
aVaX = solved_F21
aVaY = solved_F22 - 1

solved_E11 = aUaX + 0.5 * (aUaX ** 2 + aVaX ** 2)
solved_E12 = 0.5 * (aUaY+aVaX) + 0.5 * (aUaX * aUaY + aVaX * aVaY)
solved_E21 = solved_E12
solved_E22 = aVaY + 0.5 * (aUaY ** 2 + aVaY ** 2)

solved_volumetric = (solved_F11 * solved_F22) - (solved_F21 * solved_F12)
I_E = solved_volumetric - 1
trace = (solved_E11 + solved_E22) / 2
solved_dE = np.array([[solved_E11 - trace, solved_E12]
                     ,[solved_E21, solved_E22 - trace]])
solved_distortion = ((solved_E11 - trace) * (solved_E22 - trace)) - solved_E21**2
II_E = 0.5*(solved_E11 * solved_E11 + solved_E22 * solved_E22 + solved_E12 * solved_E12 + solved_E21 * solved_E21) - 0.25*I_E**2
#II_E = 0.5*(solved_dE[0,0] * solved_dE[0,0] + solved_dE[1,0] * solved_dE[1,0] + solved_dE[0,1] * solved_dE[0,1] + solved_dE[1,1] * solved_dE[1,1]) - 0.25*I_E**2
################################ Post processing ######################################
X_und = undeformed_cood[:,0]; Y_und = undeformed_cood[:,1]
X_ded = deformed_cood[:,0]; Y_ded = deformed_cood[:,1]
XX,YY = X_ded, Y_ded
# ######## surface tracking #########
# #fdm_upper = np.loadtxt("fdm_upper_E12.txt")
# #fdm_upper = np.loadtxt("fdm_upper_E11.txt")
# fdm_upper = np.loadtxt("fdm_upper_IIE.txt")

# maximum=[]; spin__ = np.zeros_like(solved_E11); upper=np.zeros_like(solved_E12); Upper=np.zeros_like(solved_E12)
# for n in np.arange(1,max(X_ded),100):
#     up=[]
#     for i in range(p_num):
#         if (n < X_ded[i]) and (X_ded[i] < (n + 100)):
#             up.append(Y_ded[i])
#     maximum.append(max(up))

# for j in range(len(maximum)):
#     for i in range(p_num):
#         if Y_ded[i] == maximum[j]:
#             upper[i] = solved_distortion[i]
#             spin__[i] = spin[i]
#             Upper[i] = 100

# fem_surface=[]
# spin_surface=[]
# temp=[]
# for i in range(len(upper)):
#     if upper[i] != 0:
#         fem_surface.append([XX[i],upper[i]])
#         spin_surface.append([XX[i],spin__[i]])
#         temp.append([XX[i],YY[i]])
# fem_surface = sorted(fem_surface,key = lambda l:l[0])

# fdm_surface=[]
# for i in range(len(fdm_upper)):
#     if fdm_upper[i] != 0:
#         fdm_surface.append([XX[i],fdm_upper[i]])

# fdm_surface = sorted(fdm_surface,key = lambda l:l[0])
# spin_surface = sorted(spin_surface,key = lambda l:l[0])
# temp = sorted(temp,key = lambda l:l[0])


# fem_surface = np.array(fem_surface, dtype=np.float64)
# fdm_surface = np.array(fdm_surface, dtype=np.float64)
# spin_surface = np.array(spin_surface, dtype=np.float64)
# temp = np.array(temp, dtype=np.float64)


# fem = fem_surface[:,1]
# fdm = fdm_surface[:,1]
# spin_upper = spin_surface[:,1]
# temp = temp[:,1]
# #fem = np.log10(np.abs(fem))
# #fdm = np.log10(np.abs(fdm))
# spin2 = np.log10(np.abs(spin))
# #############
# plt.subplot(2,1,1)
# plt.title('Shear strain profile at surface')
# plt.plot(fem, color='violet', marker='o', linestyle='--',label='IG-FEM')
# plt.plot(fdm,color='dodgerblue', marker='*',linestyle='--',label='FDM')
# plt.plot(spin2,color='blue', marker='^',linestyle='--',label='spin')
# plt.legend()
# plt.xlabel('Surface')
# plt.ylabel('distortioanl strain')

# plt.subplot(2,1,2)
# plt.title('Surface topography')
# plt.scatter(X_ded,Y_ded,c = Upper, s = rad*0.5, cmap = 'PuBuGn')
# plt.triplot(X_ded,Y_ded,ele_id, c='k', linewidth='0.3')
# plt.gca().set_facecolor('gray')
# plt.gca().set_aspect(1)


# plt.figure()
# fig, axes = plt.subplots(nrows=2, ncols=1)
# ax1,ax3 = axes.flatten()
# ax1.plot(fem, color='violet', marker='o', linestyle='--',label='IG-FEM')
# ax1.plot(fdm, color='dodgerblue', marker='*',linestyle='--',label='FDM')
# ax1.set_ylim(-4.0,0.5)
# ax2 = ax1.twinx()
# #ax2.plot(spin_upper, color='gray', marker='^',linestyle='--',label='spin')
# #ax2.set_ylim(-0.25,0.30)
# ax2.set_ylim(-0.07,0.083)
# #ax2.set_ylim(-0.08,0.09)
# plt.show()

# mid_fe_sd = []; left_fe_sd =[]; right_fe_sd =[]
# mid_spin_sd = []; left_spin_sd=[]; right_spin_sd=[]
# for i in range(p_num):
#     if 10000 < deformed_cood[:,0][i] < 15000 and 2600 < deformed_cood[:,1][i] < 3800:
#         mid_fe_sd.append(solved_E12[i])
#         mid_spin_sd.append(spin[i])
#     elif 16000 < deformed_cood[:,0][i] < 18000 and 2800 < deformed_cood[:,1][i] < 4000:
#         left_fe_sd.append(solved_E12[i])
#         left_spin_sd.append(spin[i])
#     elif 28500 < deformed_cood[:,0][i] < 31000 and 4100 < deformed_cood[:,1][i] < 5500:
#         right_fe_sd.append(solved_E12[i])
#         right_spin_sd.append(spin[i])

# sd_mid = np.sqrt(sum((mid_fe_sd - np.average(mid_fe_sd)) **2)/len(mid_fe_sd))
# rsd_mid = sd_mid/abs(np.average(mid_fe_sd))*100
# print('sd fem',sd_mid)
# print('rsd_fem',rsd_mid)

# sd_spin = np.sqrt(sum((mid_spin_sd - np.average(mid_spin_sd)) **2) / len(mid_spin_sd))
# rsd_mid_spin = sd_spin/abs(np.average(mid_spin_sd))*100
# print('sd spin',sd_spin)
# print('rsd_spin',rsd_mid_spin)

######## Circle plot #########
# strain = spin
# vmin = -0.05; vmax = 0.05
# fig=plt.figure()
# from matplotlib.colors import Normalize
# norm = Normalize(vmin, vmax)
# for i,j,r,k in zip(XX,YY,rad,strain):
#     color = plt.cm.seismic(norm(k))
#     circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.3)
#     fig.gca().add_artist(circle)
# sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='seismic', facecolors='none', vmin=vmin, vmax=vmax)
# plt.colorbar(sc)
# fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
#fig.gca().set_xlim((12600,13800)); fig.gca().set_ylim((2800,3800)) # incipiant
#fig.gca().set_xlim((28500,31000)); fig.gca().set_ylim((4100,5500)) # right
# fig.gca().set_xlim((20000,23500)); fig.gca().set_ylim((2000,4300)) # middle
#fig.gca().set_xlim((16000,18000)); fig.gca().set_ylim((2800,4000)) # left

#fig.gca().set_xlim((10000,15000)); fig.gca().set_ylim((2600,3800)) # detachment fold
#fig.gca().set_xlim((14500,17000)); fig.gca().set_ylim((3200,4200)) # fault propagation fold
#fig.gca().set_xlim((26500,29500)); fig.gca().set_ylim((4600,5600)) # fault propagation fold
# plt.gca().set_aspect(1)
# plt.gcf().set_size_inches(30, 20)
####################################
young = 12e9
poi = 0.2
lambda_constant = young*poi/((1+poi)*(1-2*poi)) #3 GPa
shear_mouli = young*(1-poi)/((1+poi)*(1-2*poi)) #13 GPa
print(lambda_constant, shear_mouli)
W = lambda_constant*0.5*((solved_E11+solved_E22)**2)+shear_mouli*(solved_E11**2+4*solved_E12+solved_E22**2)
sigma_x = young/(1-poi**2)*(solved_E11+poi*solved_E22)
sigma_xy = young*2*solved_E12/2*(1+poi)
elastic_potential_energy_11 = 0.5*lambda_constant*(solved_E11**2+solved_E22**2)+shear_mouli*(solved_E11**2)
elastic_potential_energy_12 = 0.5*lambda_constant*(solved_E11**2+solved_E22**2)+shear_mouli*(solved_E12**2)
elastic_potential_energy_22 = 0.5*lambda_constant*(solved_E11**2+solved_E22**2)+shear_mouli*(solved_E22**2)

plt.figure()
plt.scatter(XX,YY,c=elastic_potential_energy_22,s=5,cmap='inferno_r')
plt.gca().set_aspect(1)
#plt.clim(-1e10,1e10)
plt.clim(0,1e9)
plt.colorbar()
plt.show()
#print('strain energy density:',W)
#strain = np.zeros_like(solved_E11)
#strain = solved_E12#abs(solved_E11)
vmin = -1; vmax = 1
#vmin = -1; vmax = 3
fig=plt.figure()
from matplotlib.colors import Normalize
from cmcrameri import cm
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.seismic(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=3)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='seismic', facecolors='none', vmin=vmin, vmax=vmax)
plt.colorbar(sc)
#fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))

fig.gca().set_xlim((12000,14200)); fig.gca().set_ylim((2800,3800)) # incipiant
#fig.gca().set_xlim((28500,31000)); fig.gca().set_ylim((4100,5500)) # right
#fig.gca().set_xlim((16000,18000)); fig.gca().set_ylim((2800,4000)) # left
#fig.gca().set_xlim((20000,23500)); fig.gca().set_ylim((2000,4300)) # middle

#fig.gca().set_xlim((10000,15000)); fig.gca().set_ylim((2600,3800)) # detachment fold
#fig.gca().set_xlim((14500,17000)); fig.gca().set_ylim((3200,4200)) # fault propagation fold
#fig.gca().set_xlim((26500,29500)); fig.gca().set_ylim((4600,5600)) # fault propagation fold

#plt.tripcolor(deformed_cood[:,0],deformed_cood[:,1],ele_id,solved_volumetric,cmap='seismic',alpha=0.5)
plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k')
#fig.gca().set_xlim((22000,29000)); fig.gca().set_ylim((1800,3000))
plt.gca().set_aspect(1)
plt.gcf().set_size_inches(30, 20)

#############

plt.figure()
plt.tripcolor(deformed_cood[:,0],deformed_cood[:,1],ele_id,solved_volumetric,cmap='seismic',alpha=0.2)
plt.colorbar();plt.clim(-1,3)
plt.gca().set_aspect(1)
plt.show()
from cmcrameri import cm
size = 10
fig1 = plt.figure()
colormap = "inferno"#cmc.lajolla"
plt.title("new_Solving")
ax_E11 = fig1.add_subplot(2, 2, 1)
ax_E11.set_title("F11")
trip1 = plt.scatter(XX,YY,c=solved_F11, cmap=colormap,s = size)#cmc.vik
fig1.colorbar(trip1, ax=ax_E11)
ax_E11.set_aspect(1)
fig1.set_size_inches(15.5, 12.5)


ax_E12 = fig1.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX,YY,c=solved_F12, cmap="seismic",s = size)
ax_E12.set_title("F12")
fig1.colorbar(trip2, ax=ax_E12)
plt.clim(-1,1)
ax_E12.set_aspect(1)

ax_E22 = fig1.add_subplot(2, 2, 3)
trip3 = plt.scatter(XX,YY,c=solved_F22, cmap="cmc.lajolla", s = size)
ax_E22.set_title("F22")
fig1.colorbar(trip3, ax=ax_E22)
ax_E22.set_aspect(1)

ax_exact = fig1.add_subplot(2, 2, 4)
ax_exact.set_title("F21")
trip4 = ax_exact.scatter(XX, YY, c=solved_F21, cmap="cmc.lajolla",s = size)
fig1.colorbar(trip4, ax=ax_exact)
ax_exact.set_aspect(1)

fig2 = plt.figure()
plt.title("E")
ax_E11 = fig2.add_subplot(2, 2, 1)
ax_E11.set_title("E11")
trip1 = plt.scatter(XX,YY,c=solved_E11, cmap=colormap,s = size)#cmc.vik
fig2.colorbar(trip1, ax=ax_E11)
ax_E11.set_aspect(1)
fig2.set_size_inches(15.5, 12.5)
ax_E11.set_facecolor('gray')


ax_E12 = fig2.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX,YY,c=solved_E12, cmap="seismic",s = size)
ax_E12.set_title("E12")
fig2.colorbar(trip2, ax=ax_E12)
plt.clim(-8,8)
ax_E12.set_aspect(1)
ax_E12.set_facecolor('gray')

ax_E22 = fig2.add_subplot(2, 2, 3)
trip3 = plt.scatter(XX,YY,c=solved_E22, cmap=colormap, s = size)
ax_E22.set_title("E22")
plt.clim(0,20)
fig2.colorbar(trip3, ax=ax_E22)
ax_E22.set_aspect(1)
ax_E22.set_facecolor('gray')

ax_exact = fig2.add_subplot(2, 2, 4)
ax_exact.set_title("E21")
trip4 = ax_exact.scatter(XX, YY, c=solved_E21, cmap="cmc.lajolla",s = size)
fig2.colorbar(trip4, ax=ax_exact)
ax_exact.set_aspect(1)

fig3 = plt.figure()
plt.title("physical interpretation")
ax_volumetric = fig3.add_subplot(2, 2, 1)
ax_volumetric.set_title("volumetric")
trip1 = plt.scatter(XX, YY, c = solved_volumetric, cmap="seismic", s = size)#cmc.vik
fig3.colorbar(trip1, ax=ax_volumetric)
ax_volumetric.set_aspect(1)
fig3.set_size_inches(15.5, 12.5)
ax_volumetric.set_facecolor('gray')

ax_distortion = fig3.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX, YY, c = II_E, cmap='seismic', s = size)
ax_distortion.set_title("distiortion")
fig3.colorbar(trip2, ax=ax_distortion)
plt.clim(0,10)
ax_distortion.set_aspect(1)
ax_distortion.set_facecolor('gray')


fig4 = plt.figure()
plt.title("spin")
#spin_ = (spin - min(spin)) / (np.max(spin) - np.min(spin))
spin_ = spin
#norm = Normalize(-0.05, 0.05)
#color = plt.cm.rainbow(norm(spin_))
plt.scatter(XX, YY, c = abs(spin_), cmap = 'inferno', s = rad*0.2)
plt.colorbar()
plt.clim(0, 0.04)
plt.gca().set_aspect(1)
plt.gca().set_facecolor('gray')
plt.gcf().set_size_inches(15.5, 10.5)
#plt.savefig('solve_E12.png',dpi=1000)

fig5 = plt.figure()
plt.title("E11")
#plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id)
plt.scatter(XX,YY,c = solved_E11, cmap = colormap, s = rad*0.2)
plt.gca().set_aspect(1)
plt.gcf().set_size_inches(15.5, 10.5)
plt.clim(0, 1)
plt.colorbar()
plt.gca().set_facecolor('gray')
plt.gcf().set_size_inches(5.5, 5.5)

fig6 = plt.figure()
s=plt.scatter(XX, YY, c = solved_E11, s = rad * 0.25, cmap = colormap)
fig6.colorbar(s)
plt.clim(0, 1)
plt.gca().set_aspect(1)
plt.gcf().set_size_inches(5.5, 5.5)
plt.gca().set_facecolor('gray')

plt.show()