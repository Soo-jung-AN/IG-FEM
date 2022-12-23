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
spin = np.loadtxt("angular_vel0_{}.txt".format(deform))
rad = np.loadtxt("radius.txt")
############################################################################################################################################################
p_num = len(undeformed_cood)
disp = deformed_cood - undeformed_cood
u_disp = disp[:,0]; v_disp = disp[:,1]
############################################################################################################################################################
# TODO triangulation
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
trace = (solved_E11 + solved_E22) / 2
solved_dE = np.array([[solved_E11 - trace, solved_E12]
                     ,[solved_E21, solved_E22 - trace]])
solved_distortion = ((solved_E11 - trace) * (solved_E22 - trace)) - solved_E21**2
################################ Post processing ######################################
X_und = undeformed_cood[:,0]; Y_und = undeformed_cood[:,1]
X_ded = deformed_cood[:,0]; Y_ded = deformed_cood[:,1]
XX,YY = X_ded, Y_ded

strain = solved_E12
vmin = -1; vmax = 1
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
fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
plt.tripcolor(deformed_cood[:,0],deformed_cood[:,1],ele_id,strain,cmap='seismic',alpha=0.5)
plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k')
plt.gca().set_aspect(1)
plt.gcf().set_size_inches(20, 10)
#############
size = 10
fig1 = plt.figure()
colormap = "inferno"
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
trip3 = plt.scatter(XX,YY,c=solved_F22, cmap=colormap, s = size)
ax_E22.set_title("F22")
fig1.colorbar(trip3, ax=ax_E22)
ax_E22.set_aspect(1)

ax_exact = fig1.add_subplot(2, 2, 4)
ax_exact.set_title("F21")
trip4 = ax_exact.scatter(XX, YY, c=solved_F21, cmap=colormap,s = size)
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
trip4 = ax_exact.scatter(XX, YY, c=solved_E21, cmap=colormap,s = size)
fig2.colorbar(trip4, ax=ax_exact)
ax_exact.set_aspect(1)

fig3 = plt.figure()
plt.title("physical interpretation")
ax_volumetric = fig3.add_subplot(2, 2, 1)
ax_volumetric.set_title("volumetric")
trip1 = plt.scatter(XX, YY, c = solved_volumetric, cmap="seismic", s = size)
fig3.colorbar(trip1, ax=ax_volumetric)
ax_volumetric.set_aspect(1)
fig3.set_size_inches(15.5, 12.5)
ax_volumetric.set_facecolor('gray')

ax_distortion = fig3.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX, YY, c = solved_distortion, cmap=colormap, s = size)
ax_distortion.set_title("distiortion")
fig3.colorbar(trip2, ax=ax_distortion)
plt.clim(0,10)
ax_distortion.set_aspect(1)
ax_distortion.set_facecolor('gray')

plt.show()