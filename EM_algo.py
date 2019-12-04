import numpy as np
import math
from PIL import Image
from tempfile import TemporaryFile
import multiprocess as mp
import itertools
from scipy import fftpack
import cv2
import os
from PIL import ImageFilter
import skimage
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal


def shift_rows_and_cols( arr, i, j_1, rows, cols, N):
    #import pdb; pdb.set_trace()  
    arr = np.roll(arr, i, axis = 1)  # shift left(-ve) or right(+ve) by ith
   
    arr = np.roll(arr, j_1, axis = 0)  # shift up(-ve) or down(+ve) by jth

    arr = arr[N:N+rows, N:N+cols]    
    return arr

def create_shifted_version_of_F(F_zero_pad, n, rows, cols):
    
    rows_z, cols_z = F_zero_pad.shape[0], F_zero_pad.shape[1]
    F_shift = np.zeros((rows,cols,n*n)) 
     
    N = int((n-1)/2) 
    i = -N
    count_r = 0 
    
    while  i <= N:
          j_1 = -N
          while j_1 <= N:
                
               F_shift[:, :,count_r] =  shift_rows_and_cols(F_zero_pad, i, j_1, rows, cols, N) 
               count_r = count_r + 1
               j_1 =  j_1 + 1
          i = i + 1
  
    #import pdb; pdb.set_trace()
    #F_shift = F_shift[N:N + rows, N : N + cols]  
    return F_shift

def calculate_C_matrix(F_shift, W, n):
    
    C  = np.zeros((n*n,n*n)) 
    new_rows, new_cols = n*n -1, n*n -1
    C_new = np.zeros(( new_rows, new_cols ))
    N = int((n-1)/2) 
    
    col_del = int(((n*n) +1)/2 - 1)
    row_del = int(((n*n)+1)/2  - 1)

    for i in range(n*n):
        for j in range(n*n):
          
          C[i, j] = np.sum(np.multiply(np.multiply(W, F_shift[:, :, i]), F_shift[:,:,j]))
   
     
    C_new = np.delete(np.delete(C, col_del, axis = 1), row_del, axis=0)
    return C_new


def calculate_R_matrix(F_shift, F, W, n):
    
    R = np.zeros(n*n)
    N = int((n-1)/2) 
    
    new_row = n*n - 1
    
    R_new = np.zeros(new_row)
    col_del = int(((n*n) +1)/2 - 1)

    for i in range(n*n):
       
         R[i] = np.sum(np.multiply(np.multiply(W, F_shift[:, :, i]), F))

    #import pdb ; pdb.set_trace()
    R_new = np.delete(R, col_del)
    
    
    return R_new

 
 
def one_time_calculation(F_zero_pad, n , rows, cols, img_path):
  
      
     rows_zero_pad, cols_zero_pad = F_zero_pad.shape[0], F_zero_pad.shape[1]
     
     N = int((n-1)/2)

     f_sub = np.zeros((n, n, rows, cols))
     

     '''
     # parrellizing the process
     p = mp.Pool()
     
     def fun(F_zero_pad, N):
           
        
     i = N
     input = ((i,j) for i, j in itertools.combinations_with_replacement(xrange(N, cols + N),2) if  i  < rows + N)

           = p.map(fun, input)

     '''
     c_r = 0
     for i in range(N,  rows+N): 
        c_j = 0
        for j in range(N,  cols+N):
        
            f_sub[:, :, c_r, c_j]  = F_zero_pad[i-N : i+N+1, j-N : j+N+1]

            #print(i,j)
            
            c_j = c_j + 1
        c_r = c_r + 1 
  
    
     r = str(rows)
     c = str(cols)
     save_file = './numpy_files/'+img_name +'_' + r + c + '.npy'
     np.save(save_file, f_sub)
     
     


def calculate_summation_term(F_zero_pad, alpha, n, rows, cols, img_name):
   
      
     rows_zero_pad, cols_zero_pad = F_zero_pad.shape[0], F_zero_pad.shape[1]
    
    
     r = str(rows)
     c = str(cols)#img_G = img_G[0::2, 0::2]
     read_path  = './numpy_files/'+img_name +'_' +r + c +'.npy'
     f_sub = np.load(read_path)
     N = int((n-1)/2)
 
     #F_out = np.zeros((rows_zero_pad, cols_zero_pad))
     F_out = np.zeros((rows, cols))

    
     # Parrallize the process
      
     count = 0
     #def fun(f_sub):
     #    F_out = np.zeros((rows, cols))
 
     #    F_out[i, j] = np.sum(np.matmul(alpha, f_sub[:,:,i,j].T))
     #    print (count)
     #    count = count +1 
     #    return F_out
        
 
     #p = mp.Pool()
     #input = ((i ,j) for i,j in itertools.combinations_with_replacement(range(cols),2) if i < rows)
     
    # F_out = p.map(fun, input)
     #import pdb; pdb.set_trace()
     #p.close()
    # p.join()
    
      
     alpha_flat = alpha.flatten()    
     for i in range(0, rows):
         
         for j in range(0, cols):
              
              f_sub_flatten = f_sub[:,:,i,j].flatten()
              F_out[i, j] = np.matmul(alpha_flat, f_sub_flatten)
    
              #print(i, j)        
      
                  
     
     
     #F_out_final = F_out[N:(rows_zero_pad - N), N: (cols_zero_pad - N)]
     
     F_out = F_out / 255 
     return F_out
 
def EM_Algo(F, n, threshold, img_name):
    
    alpha = np.random.random_sample((n,n)) # intialize alpha
    sigma = 0.0075     # intialize variance

    
    
    p_0   = (1/256)   # reciprocal to the range of the image range
     
    rows, cols = F.shape[0], F.shape[1]
     
    N = int((n-1)/2)
  
    new_ele_pos = int(((n*n)+1)/2 -1)
  
    alpha[N, N] = 0        # set alpha[0,0] = 0, given in paper
    F_zero_pad = np.pad(F, N)
    rows_zero_pad, cols_zero_pad = F_zero_pad.shape[0], F_zero_pad.shape[1]
    
       
    residual = np.zeros((rows, cols))
    P = np.zeros((rows, cols))
    W = np.zeros((rows, cols))
    
    F_shift = create_shifted_version_of_F(F_zero_pad, n, rows, cols)
   
    iteration = 0 

    para     = str(n)
    save_alpha  ='./numpy_files/' + img_name + para + 'alpha.npy'
    save_p_map  = './numpy_files/'+ img_name + para + 'prob_map.npy'


    one_time_calculation(F_zero_pad, n , rows, cols, img_name)
  
    sigma_old = sigma 
    alpha_old = alpha
    
    
    while(True): 
     
            # Expectation step

            F_out_final = calculate_summation_term(F_zero_pad, alpha_old, n, rows, cols, img_name)   
            residual  = np.abs(F - F_out_final) 
        
            P = (1 / (sigma_old * np.sqrt(2 * np.pi))) * np.exp(- (residual ** 2) / ( 2 * (sigma_old ** 2)))        # Conditional Probability 
    
            W = P / (P + p_0)   # posterior probability
   
            # Maximization step
    
            C  = calculate_C_matrix(F_shift, W,  n)
   
            #C [n,n] = 0
            R  = calculate_R_matrix(F_shift, F, W, n)
            
            # update alpha

            alpha_new = np.matmul(np.linalg.inv(C), R.T)

            # insert zero
            alpha_new = np.insert(alpha_new, new_ele_pos, 0)
            
            alpha_new = alpha_new.reshape((n,n)) 
       
        
            #alpha[N, N] = 0
            # update variance
    
            sigma_new = np.sqrt(np.sum(np.multiply(W, residual ** 2)) / np.sum(W))
            error   = np.sum(np.abs(alpha_new - alpha_old))
         
            sigma_old = sigma_new 
            alpha_old = alpha_new
            print(error)
            if (error < threshold):     # stopping condition
                 break
  

    return alpha_old, P

def DFT_each_block(prob_map,block_size, img_name ):
    
     k = np.array([[-1/4, 1/2, -1/4],[1/2,-1,1/2],[-1/4, 1/2, -1/4]])
     
     result  = '/home/dkjangid/Desktop/PhD/Fall19/DIP/Project/Image_Forensic/Image_Forensic/results' 
     rows, cols = prob_map.shape[0], prob_map.shape[1]
    
     plt.imsave('{}_prob_map.jpg'.format(img_name), prob_map, cmap = matplotlib.cm.gray)    
     zero_pad_r = block_size - (rows % block_size)
     zero_pad_c  = block_size - (cols % block_size)
     prob_map_new = np.zeros((rows+zero_pad_r, cols+zero_pad_c))
     prob_map_new[0:rows, 0:cols] = prob_map
    
     rows_new, cols_new = prob_map_new.shape[0], prob_map_new.shape[1]
     no_of_blocks = (rows_new*cols_new) // (block_size * block_size)
     prob_map_block = np.zeros((block_size, block_size, no_of_blocks))

     
     import pdb; pdb.set_trace()

     count = 0
     for i in range(rows // block_size):
         
         for j in range(cols // block_size):
             
             prob_map_block[:,:,count]   = prob_map[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)]
             count = count + 1
 

     #prob_map_block = np.reshape(prob_map_new, (block_size, block_size, no_of_blocks), order='F')

     import pdb; pdb.set_trace()  
     for count in range(no_of_blocks): 
                        


           plt.imsave(result+'/{}_orgimg/'.format(img_name)+img_name+"_{}.jpg".format(count),prob_map_block[:,:,count], cmap = matplotlib.cm.gray, vmin=0, vmax=1)

           prob_map_1_1 = signal.convolve2d(prob_map_block[:,:,count],k, mode='full')
 
           f = 100 
           prob_map_fft_1 = fftpack.fft2(prob_map_1_1)
           prob_map_fft_1 = np.fft.fftshift(prob_map_fft_1)
           prob_map_fft_1_1 = 20* np.log( np.abs(prob_map_fft_1))
     
           # High Pass Filter
           prob_map_fft_1_hp = prob_map_fft_1_1
           #prob_map_fft_1_hp = np.zeros((2*rows, 2 * cols))
           prob_map_fft_1_hp[cols-f:cols+f, rows-f:rows+f] = 0

      
           # Blur 
      
           prob_map_fft_1_blur =  skimage.filters.gaussian(prob_map_fft_1_hp, sigma= (5,5))
           prob_map_fft_1_1_scaled = prob_map_fft_1_blur/ np.max(prob_map_fft_1_blur)   

           # Thresholding
     
           prob_map_fft_1_1_scaled[prob_map_fft_1_1_scaled>0.98] = 1
           prob_map_fft_1_1_scaled[prob_map_fft_1_1_scaled<=0.98]= 0
           # Gamma Correction
     
           gamma = 2
           prob_map_fft_1_final = np.array(255*(prob_map_fft_1_1_scaled) ** gamma, dtype ='uint8')

           plt.imsave(result+'/{}_fft/'.format(img_name)+img_name+ "fft_{}.jpg".format(count), prob_map_fft_1_final, cmap = matplotlib.cm.gray) 
           
     
if __name__ == "__main__":   
     n  = 3   # no of parameters
     threshold = 0.001
   
     img_path = '/home/dkjangid/Desktop/PhD/Fall19/DIP/Project/Image_Forensic/Image_Forensic/data/8.jpg'  
   
      
     import pdb; pdb.set_trace()
  

     img  = cv2.imread(img_path)
     img = img[:,:,1]

     img_name = os.path.basename(img_path).replace('.jpg', '')
 
     # take only one channel
     #img_R = img[:, :, 1]
  
    
  
     #dim = (640, 480)
   
     rows_img, cols_img = img.shape[0], img.shape[1]
     dim = (int(cols_img/2), int(rows_img/2))
     img_G = img[0::2, 0::2]
     img_G = img_G/255
     #img_G = cv2.resize(img_G, dim)
     #import pdb ; pdb.set_trace()
     
     
     para   = str(n)
     save_alpha  ='./numpy_files/' + img_name + para + 'alpha.npy'
     save_p_map  = './numpy_files/'+ img_name + para + 'prob_map.npy'

     
     
     alpha, prob_map  = EM_Algo(img_G, n, threshold, img_name)
     
     #import pdb; pdb.set_trace() 
     print (alpha)
     prob_map = prob_map/np.sum(prob_map)
     
     np.save(save_alpha, alpha)
     np.save(save_p_map, prob_map)
    
     alpha = np.load(save_alpha)
     prob_map = np.load(save_p_map)

     
     prob_map = prob_map/np.amax(prob_map) 
      
     ''' 
     prob_map[prob_map>0.4] = 1
     prob_map[prob_map<=0.4] = 0
     
     prob_map = prob_map * 255
     '''
     #prob_map = np.uint(prob_map * 255)
    
     rows, cols = prob_map.shape[0], prob_map.shape[1]

     # p_map upsampled by factor of 2
     
     prob_map = cv2.resize(prob_map, (2*cols, 2*rows), interpolation = cv2.INTER_AREA)
     
     #prob_map[prob_map>0.5] = 1
     #prob_map[prob_map<=0.5] = 0

     DFT_each_block(prob_map, 300, img_name)

      

