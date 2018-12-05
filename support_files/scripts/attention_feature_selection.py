import numpy as np
import math
import matplotlib.pyplot as plt
import pprint
###############################################################################
################ FUNCTION DESCRIPTIONS #######################################
# Future_rotation: returns the predicted rotation for future states for the time
# horizon based on the current R_k, omega_k at time k
#
# Future_poses: returns the predicted future states for the time horizon based on
# the IMU data at time k
#
# Landmark_est_from_PGO: TODO placeholder dummy function at the moment, should query
# PGO to get pose estimate of landmarks in np.array with dimensions landmarks on rows,
# and x,y,z of each landmark position in cols
#
# Visibility_check: returns the projected pixel positions of the landmarks in the
# predicted future states along with a flag in the third dimension of u_pix which
# indicates if it is still in the frame (visible) or
#
# Cov_current_state: TODO placeholder dummy function at the moment, should query
# PGO to get covariance of the current state
#
# Inf_mat_no_features: returns omega_k_kH (the information matrix of the future states
# assuming no landmark information)
#
# Inf_mat_features: returns del_l (information matrix of the future states associated
# with each l'th feature (features along depth dimension))
#
# UpperBounds: calculates upperbounds on each feature NOTE: definitely looks suspect,
# since domain going 0 or negative for log often
#
# F_value: calculates objective function for each feature NOTE: definitely looks
# suspect, since domain going 0 or negative for log often
#
# Attentive_feature_selection: Runs greedy algorithm and returns selected subset
###############################################################################
############################### PARAMETERS#####################################
# rotation matrix at current time k
R_k = np.asarray([[1, 0, 0],[0,1,0],[0,0,1]]) # TODO: PLACEHOLDER, replace with PGO
# angular velocity at current time k
omega_k = 1.1 # TODO: PLACEHOLDER, replace with IMU
# current state estimate at time k, contains t_k, v_k, and b_k
x_k = np.asarray([1.0, 1.0, 1.0, 3.0, -3.0, 3.0, 0.01, 0.01, 0.01]) # TODO: PLACEHOLDER, replace with PGO
# acceleration measurement
a_k = np.asarray([1.0, -1.0, 0.01]) # TODO: PLACEHOLDER, replace with IMU
# sensor noise
eta_k = np.asarray([0.005, 0.005, 0.005]) # TODO: PLACEHOLDER, guess
# random vector of IMU bias
eta_kj_b = np.asarray([0.001, 0.001, 0.001]) # TODO: PLACEHOLDER, guess
# sampling time of IMU
delta = 1.0/30.0 # TODO: PLACEHOLDER, replace with delta between frames
# time horizon
H = 5 # tunable parameter
# intrinsic camera calibration matrix
camera_calibration = np.asarray([[1,2,5],[4,5,6], [7,8,9]]) # TODO: PLACEHOLDER, replace with real camera info
# image dimensions (number of pixels length x width)
image_dimension = (1,640,1,480) # TODO: PLACEHOLDER, replace with real camera info
# covariance of IMU
IMU_cov = 0.01*np.identity(3) # TODO: PLACEHOLDER,guess
# variance of IMU
IMU_var = 0.1 # TODO: PLACEHOLDER, guess
# rotation of IMU w.r.t camera
R_IMU_cam = np.identity(3) # TODO: PLACEHOLDER, replace with real camera info
# set of total features to choose from
feature_set = np.linspace(1,6,6) # TODO: PLACEHOLDER, do not change unless we first fix Landmark_est_from_PGO() function
# probability of tracking the l'th feature
p_l = np.ones(6)*0.7 # TODO: PLACEHOLDER, do not change length unless we first fix Landmark_est_from_PGO() function
# flag to indicate which cost function to use
flag = 'min_eig' # can switch between 'log_det' and 'min_eig'
# number of features to select for subset
K = 3 # number of features to select; do not change until we first fix Landmark_est_from_PGO() function
##############################################################################
def Future_rotation(R_k,H,omega_k, delta):
    # assuming angular velocity constant, forward integrate the rotation
    R = np.zeros((3*H,3))
    old_R = R_k # initialize old_r as R at time k
    for j0 in range(0,H):
        new_R = np.add(omega_k*delta*(j0+1),old_R)
        R[j0*3:(j0*3+3),:] = new_R
        old_R = new_R
    return R
def Future_poses(R, x_k, a_k, eta_k, eta_kj_b, delta, H):
    v = np.zeros((3*H)) # velocity [vx, vy, vz]
    t = np.zeros((3*H)) # position [tx, ty, tz]
    b = np.zeros((3*H)) # noise [bx, by, bz]
    g = np.asarray([0, -9.8, 0]) # gravity
    t_k = x_k[0:3] # extract t_k from x_k state vector
    v_k = x_k[3:6] # extract v_k from x_k state vector
    b_k = x_k[6:] # extract b_k from x_k state vector
    # simulate forward future velocities
    for j in range(1,H+1):
        step2 = np.zeros((3)) # initialize
        for i in range(0,j-1): # assuming k = 0
            step1 = np.matmul(R[i*3:(i*3+3),:],np.add(a_k,np.add(-b_k,-eta_k))*delta) # TODO: assuming acceleration, eta constant, a[i], eta[i] in paper?
            step2 = np.add(step2,step1)
        v[(j-1)*3:((j-1)*3+3)] = np.add(g*delta*j,np.add(v_k,step2))
    # simulate forward future positions
    for j in range(1,H+1):
        step3 = np.zeros((3)) #initialize
        for i2 in range(0,j-1): # assuming k = 0
            step1 = np.matmul(R[i2*3:(i2*3+3),:],np.add(a_k,np.add(-b_k,-eta_k)))*(delta**2)*0.5 # TODO: assuming acceleration, eta constant, a[i], eta[i] in paper?
            step2 = np.add(v[i2*3:(i2*3+3)]*delta,np.add(step1, 0.5*g*delta**2))
            step3 = step3 + step2
        t[(j-1)*3:((j-1)*3+3)] = np.add(t_k, step3)
    # simulate forward future noise
    for j in range(0,H):
        b[j*3:(j*3+3)] = b_k + eta_kj_b
    x_k2H = []
    # merge future position, velocity, and noise into future states vector
    for j in range(0,H):
        x_k2H.append(t[j*3:(j*3+3)])
        x_k2H.append(v[j*3:(j*3+3)])
        x_k2H.append(b[j*3:(j*3+3)])
    return x_k2H,t,v,b # future state vector from time k to k + H
################# TESTING FOR Future_poses FUNCTION ###########################
# future_states,t,v,b = Future_poses(R, x_k, a_k, eta_k, eta_kj_b, delta, H) # list of numpy arrays
# print(future_states)
# v_x = v[0::3]
# v_y = v[1::3]
# v_z = v[2::3]
# time = np.linspace(1,H,H)
# print(time)
# print(v_x.shape)
# plt.plot(time,v_x,'g', time, v_y,'b', time, v_z,'r')
# plt.show()
##################### END TESTING ##############################################
def Landmark_est_from_PGO():
    ### TODO pipe from PGO optimization in VINS-MONO all the landmark estimates
    ### for now, return dummy variable
    feature_estimates = np.asarray([[0,0,0],[0,0.2,0],[10,0,0.2],[3,3,3], [2,1,7],[0,-0.2,0]]) # features along row, x,y,z along col
    return feature_estimates

def Visibility_check(R, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, feature_set,image_dimension):
    future_states,t,v,b = Future_poses(R, x_k, a_k, eta_k, eta_kj_b, delta, H)
    # extract camera information
    u_min = image_dimension[0]
    u_max = image_dimension[1]
    v_min = image_dimension[2]
    v_max = image_dimension[3]
    sx_and_f = float(camera_calibration[0,0])
    sy_and_f = float(camera_calibration[1,1])
    o_x = camera_calibration[0,2]
    o_y = camera_calibration[1,2]
    # extract position along axes vectors
    t_x = t[0::3]
    t_y = t[1::3]
    t_z = t[2::3]
    # initialize u and v vectors to hold pixels
    # NOTE: u_pix contains third dimension to hold visibility check (1 = visible, 0 = not visible)
    u_pix = np.zeros((H,len(feature_set),2)) # only indexing if visibile in one of the arrays
    v_pix = np.zeros((H,len(feature_set)))
    # for all future frames, calculate predicted pixel projection for each feature
    for j in range(0, H): # for each position in a future frame
        t_vec = np.asarray([t_x[j], t_y[j], t_z[j]])
        t_mat = np.asarray([[t_x[j], t_x[j], t_x[j]], [t_y[j], t_y[j], t_y[j]], [t_z[j], t_z[j], t_z[j]]]) # make sure vertical vec
        p_lo = Landmark_est_from_PGO()
        # iterate over each feature for every future frame and check visibility
        for f in range(0,len(feature_set)):
            p_lc = np.add(p_lo[f,:], np.transpose(-t_vec))
            u_pix[j,f,0] = sx_and_f*(p_lc[0].item()/p_lc[2].item()) + o_x
            u_pix_scalar = u_pix[j,f,0]
            u_pix_scalar = u_pix_scalar.item()
            v_pix[j,f] = sy_and_f*(p_lc[1].item()/p_lc[2].item()) + o_y
            v_pix_scalar = v_pix[j,f]
            v_pix_scalar = v_pix_scalar.item()
            if u_pix_scalar < u_max and u_pix_scalar > u_min and v_pix_scalar < v_max and v_pix_scalar > v_min:
                u_pix[j,f,1] = 1
    return u_pix, v_pix,future_states,t,v,b,R

#################### COMMENT BACK IN FOR TESTING, PRINTING, AND PLOTTING #######
# u_pix, v_pix,future_states,t,v,b,R = Visibility_check(R, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, feature_set, image_dimension)
# print(u_pix)
# # print(v_pix.shape)
# print(v_pix)
# plt.plot(time,u_pix[:,1,0],'g')
# plt.show()
# plt.plot(time,v_pix[:,1])
# plt.show()
# # TODO: plot with different color if out of the frame
################################################################################

def Cov_current_state():
    # TODO: make a random 9x9 symmetric pos. semi-def matrix and pass that as the covariance at time k
    cov = np.identity(9)*0.1
    inf = np.linalg.inv(cov)
    return inf

def Inf_mat_no_features(H, delta, R, IMU_bias_cov, IMU_var):
    # prior IMU information matrix (0 everywhere except top left block)
    inf_mat_prior = np.zeros((9*(H+1),9*(H+1)))
    inf_mat_prior[0:9,0:9] = Cov_current_state()
    # calculate IMU infomration matrix for future poses
    # Steps: 1) Calculate A_kj matrix
    #        2) Calculate information matrix between k,j from IMU
    #        3) Multiply and sum
    A_kj = np.zeros((9*H, 9*(H+1),H)) # stacking A_kj in Eq. 50; rows = 9*H, 9 for each kj pair in k'th iteration,
                                                            # col = 0*H cols from formula, for each kj pair in k'th iteration
                                                            # depth = k iterations
    IMU_cov_kj = np.zeros((9*H,9*H)) # initalize
    IMU_inf_kj = np.zeros((9*H,9*H)) # initialize
    # calculating over all future frames
    for k in range(0,H):
        N_kj = 0
        M_kj = 0
        # calculate over all k-j pairs of future frames where j > k
        for j in range(k+1,H):
            # calculate CC^T (sub_block of inf_mat_kj)
            CC_T_block1 = np.zeros((3,3))
            CC_T_block23 = np.zeros((3,3))
            CC_T_block4 = (j-k-1)*delta**2*np.identity(3)
            for i in range(k,j):
                # calculate N, M (sub-blocks of A_kj)
                N_kj = N_kj + (j-i-0.5)*R[i*3:(i*3+3),:]*delta**2
                M_kj = M_kj + R[i*3:(i*3+3),:]*delta
                CC_T_block1 = np.add(CC_T_block1,(((j - i -0.5)**2)*delta**4*np.identity(3)))
                CC_T_block23 = np.add(CC_T_block23,((j-i -0.5)*delta**3*np.identity(3)))
            # stack to make CC^T
            CC_T = np.vstack((np.hstack((CC_T_block1,CC_T_block23)),np.hstack((CC_T_block23, CC_T_block4))))
            # calculate IMU information matrix for kj
            IMU_cov_kj[k*9:k*9+ 9,j*9:j*9 + 9] = np.vstack((np.hstack((np.multiply(IMU_var,CC_T),np.zeros((6,3)))), np.hstack((np.zeros((3,6)), IMU_bias_cov))))
            IMU_inf_kj[k*9:k*9+ 9,j*9:j*9 + 9] = np.linalg.inv(IMU_cov_kj[k*9:k*9 + 9,j*9:j*9 + 9])
            # caluclate sub-block to make A_kj
            A_block = np.multiply(np.identity(9),-1.0)
            A_block[0:3,3:6] = np.multiply(np.identity(3),-delta*(j-k))
            A_block[0:3,6:] = N_kj
            A_block[3:6,6:] = M_kj
            A_kj[j*9:j*9 + 9,j*9:j*9+9,k] = A_block
            A_kj[j*9:j*9 + 9,j*9+9:j*9+18,k] = np.identity(9)

    # Step 3 (multiply and sum to find information matrix without features)
    inf_mat_IMU_k2kH = np.zeros((9*(H+1),9*(H+1)))
    for k in range(0,H):
        for j in range(k+1,H): # TODO: or in range k to H?
            inf_mat_IMU_k2kH = np.add(inf_mat_IMU_k2kH,np.matmul(np.transpose(A_kj[j*9:j*9+9,:,k]), np.matmul(IMU_inf_kj[k*9:k*9 + 9,j*9:j*9 + 9],A_kj[j*9:j*9+9,:,k])))
    test_eigs, test_eigv = np.linalg.eig(inf_mat_IMU_k2kH)
    print('#######')
    print(np.amin(test_eigs))
    inf_mat_k2kH = np.add(inf_mat_IMU_k2kH,inf_mat_prior)
    return inf_mat_k2kH
###################COMMENT BACK IN FOR TESTING AND PRINTING####################
# omega_k_kH_test = Inf_mat_no_features(H, delta, R, IMU_cov, IMU_var) ## TODO: watch, has same variable as in attention main function
# print(omega_k_kH_test)
###############################################################################
def Inf_mat_features(R, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, feature_set, image_dimension, R_IMU_cam):
    u_pix,v_pix,future_states,t,v,b,R = Visibility_check(R, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, feature_set, image_dimension)
    # TODO: Once you figure out what u_kl is, then we can code up F, and E, and the other equations
    u_kl = np.zeros((H,len(feature_set),3)) # depth = 3 components of u-vec + whether it's visible, width = different features, len = different k's
    del_l = np.zeros((9*(H+1),9*(H+1),len(feature_set)))
    for f in range(0,len(feature_set)):
        flag_seen = False
        # print('iterating again')
        i_still_in_frame = 0 # (re)initialize counter
        F = np.zeros((3*np.count_nonzero(u_pix[:,f,1]),9*(H+1)))
        E = np.zeros((3*np.count_nonzero(u_pix[:,f,1]),3))
        for i in range(0,H):
            u_kl[i,f,0] = u_pix[i,f,0]
            u_kl[i,f,1] = v_pix[i,f]
            u_kl[i,f,2] = 1
            if u_pix[i,f,1] != 0:
                flag_seen = True
                # print('still visible')
                u_kl_skew = np.asarray([[0, -u_kl[i,f,2], u_kl[i,f,1]],[u_kl[i,f,2],0,u_kl[i,f,0]],[-u_kl[i,f,1],u_kl[i,f,0],0]])
                F[3*i_still_in_frame:3*i_still_in_frame+3,9*i:9*i+3] = np.matmul(u_kl_skew,np.transpose(np.matmul(R[i],R_IMU_cam)))
                E[3*i_still_in_frame:3*i_still_in_frame+3,:]=np.matmul(u_kl_skew,np.transpose(np.matmul(R[i],R_IMU_cam)))*(-1)
                i_still_in_frame = i_still_in_frame + 1
        # TODO what to do when no features detected at all
        if flag_seen == True:
            # print('DET MATRIX')
            # print(np.matmul(np.transpose(E[:,:]),E[:,:]))
            # print('DETERMINANT##########    ')
            # print(np.linalg.det(np.matmul(np.transpose(E[:,:]),E[:,:])))
            step1 = np.matmul(np.transpose(F[:,:]),E[:,:])
            step2 = np.matmul(step1, np.linalg.inv(np.matmul(np.transpose(E[:,:]),E[:,:])))
            step3 = np.matmul(step2, np.matmul(np.transpose(E[:,:]),F[:,:]))
            del_l[:,:,f] = np.add(np.matmul(np.transpose(F[:,:]),F[:,:]), -1*step3)
        else:
            del_l[:,:,f] = np.zeros((9*(H+1),9*(H+1)))
    return del_l # del_l format: l'th feature in depth

def UpperBounds(feature_subset,feature_set, omega_k_kH,del_l, p_l, H, flag):
    ############## TODO need to fix.
    # M = np.zeros((9*(H+1),9*(H+1)))
    # print(omega_k_kH)
    upper_bound = np.zeros((len(feature_set)))
    if flag == 'min_eig': ## we only need to do this once for min_eig case
        eigvals, eigvecs = np.linalg.eig(omega_k_kH) ## what type is this...?
        min_eig = np.amin(eigvals)
        min_arg = np.argmin(eigvals)
        min_eigenvec = eigvecs[:,min_arg]
    if flag == 'log_det':
        for f in range(0,len(feature_set)): # f = feature_index
            sum_del_l_inf_mat = np.zeros((9*(H+1),9*(H+1))) # reinitialize
            M = np.zeros((9*(H+1),9*(H+1))) # reinitialize
            feature_number = f + 1
            proposed_subset = feature_subset.copy()
            for p_f in proposed_subset: # p_f = feature number
                p_f_index = p_f - 1
                sum_del_l_inf_mat[:,:] = np.add(sum_del_l_inf_mat,p_l[p_f_index].item()*del_l[:,:,p_f_index])
            M[:,:] = np.add(omega_k_kH,sum_del_l_inf_mat)
            ######### same for F_value func up til here #######################
            for row_col in range(0,9*(H+1)):
                if M[row_col,row_col] > 0:
                    upper_bound[f] = upper_bound[f] + math.log(M[row_col,row_col])
    if flag == 'min_eig':
        for f in range(0, len(feature_set)):
            upper_bound[f] = min_eig + np.linalg.norm(np.matmul(del_l[:,:,f],min_eigenvec))
    return upper_bound

def F_value(feature_subset,omega_k_kH,del_l, p_l, H, flag):
    for f in range(0,len(feature_subset)): # f = feature_index
        sum_del_l_inf_mat = np.zeros((9*(H+1),9*(H+1))) # reinitialize
        M = np.zeros((9*(H+1),9*(H+1))) # reinitialize
        sum_del_l_inf_mat[:,:] = np.add(sum_del_l_inf_mat,p_l[f].item()*del_l[:,:,f])
    M[:,:] = np.add(omega_k_kH,sum_del_l_inf_mat)
    if flag == 'log_det':
        determinant_M = np.linalg.det(M)
        if determinant_M > 0:
            f_val = math.log(determinant_M)
        else:
            print('feature_subset without valid determinant')
            print(feature_subset)
            f_val = float("inf")*-1
        return f_val
    if flag == 'min_eig':
        eigvals, eigvecs = np.linalg.eig(M)
        f_val = np.amin(eigvals)
        return f_val

def Attentive_feature_selection(R_k, omega_k, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, image_dimension, IMU_cov, IMU_var, R_IMU_cam, feature_set, p_l, flag, K):
    R = Future_rotation(R_k,H,omega_k, delta)
    omega_k_kH = Inf_mat_no_features(H, delta, R, IMU_cov, IMU_var)
    del_l = Inf_mat_features(R, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, feature_set, image_dimension, R_IMU_cam)
    subset = set() # initialize subset list
    N = len(feature_set) # number of landmarks
    for i in range(0,K):
        # compute upper bound for cost function used
        upper_bounds = UpperBounds(subset, feature_set, omega_k_kH, del_l, p_l, H, flag) # numpy_array, add upper bound for each inf. matrix associated with each landmark
        upper_bounds_desc_ind = np.argsort(upper_bounds,axis=0) # sort array in descending order of upper-bounds TODO: check axis...
        # initialize best feature
        f_max = -1
        l_max = -1
        print('upper_bounds_desc_ind')
        print(upper_bounds_desc_ind)
        print('upper bounds')
        print(upper_bounds)
        for ub in np.nditer(upper_bounds_desc_ind): # over all sorted features, pick greedily
            if upper_bounds[ub] < f_max: # if f_max better than current best upper bound, stop looking (since list is sorted)
                break
            proposed_subset2 = subset.copy()
            proposed_subset2.add(ub+1)
            new_f = F_value(proposed_subset2, omega_k_kH, del_l, p_l, H, flag)
            if new_f > f_max:
                f_max = new_f
                l_max = ub+1 ## TODO: decide on +1 or just index features using 0 as well...
        subset.add(l_max)
    return subset
subset = Attentive_feature_selection(R_k, omega_k, x_k, a_k, eta_k, eta_kj_b, delta, H, camera_calibration, image_dimension, IMU_cov, IMU_var, R_IMU_cam, feature_set, p_l, flag, K)
print(subset)
