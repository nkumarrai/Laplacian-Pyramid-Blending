# CSE 527 - HW 1
# Submitted By - Naveen Kumar Rai
# Student Id - 111207633

# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):
   im = img_in
   im_shape = im.shape

   if len(im_shape) < 3:
      im = im[:,:,np.newaxis]

   (h, w, d) = im.shape

   if type(im[0,0,0]) == np.dtype(np.float32) or type(im[0,0,0]) == np.dtype(np.float16):
      im = np.array(im*255).astype('uint8')
   else:
      im = np.array(im).astype('uint8')

   out = np.zeros(im.shape)
   out_inbuilt = np.zeros(im.shape)
   for i in range(d):
      frame = im[:,:,i]
      hist_frame = cv2.calcHist([frame],[0],None,[256],[0,256])
      cdf = np.cumsum(hist_frame)
      cdf_normalized = cdf * hist_frame.max()/ cdf.max()
      cdf_m = np.ma.masked_equal(cdf,0)
      cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
      cdf = np.ma.filled(cdf_m,0).astype('uint8')
      frame_equalized = cdf[frame]
      cv2.imshow("equalized", frame_equalized)
      out[:,:,i] = frame_equalized

   if len(im_shape) < 3:
      out = out[:,:,0]
      im = im[:,:,0]
   
   img_out = out
   img_out = np.array(img_out).astype('uint8')
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   # output_name = sys.argv[3] + "1.jpg"
   output_name = sys.argv[3] + "output1.png"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
   img = img_in
   if len(img.shape) >= 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   f = np.fft.fft2(img)
   fshift = np.fft.fftshift(f)
   magnitude_spectrum = 20*np.log(np.abs(fshift))

   rows = img.shape[0]
   cols = img.shape[1]
   crow,ccol = rows/2 , cols/2
   mask = np.zeros(fshift.shape)
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1
   fshift = fshift * mask

   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_back = np.abs(img_back)

   img_out = img_back
   img_out = np.array(img_out).astype('uint8')
   return True, img_out

def high_pass_filter(img_in):
   img = img_in
   if len(img.shape) >= 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   f = np.fft.fft2(img)
   fshift = np.fft.fftshift(f)
   magnitude_spectrum = 20*np.log(np.abs(fshift))

   rows = img.shape[0]
   cols = img.shape[1]
   crow,ccol = rows/2 , cols/2
   fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_back = np.abs(img_back)

   img_out = img_back
   img_out = np.array(img_out).astype('uint8')
   return True, img_out
   
def deconvolution(img_in):
   blurred = img_in
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T

   def ft(im, newsize=None):
       dft = np.fft.fft2(np.float32(im),newsize)
       return np.fft.fftshift(dft)

   def ift(shift):
       f_ishift = np.fft.ifftshift(shift)
       img_back = np.fft.ifft2(f_ishift)
       return np.abs(img_back)

   #Deconvolution
   imf = ft(blurred, (blurred.shape[0],blurred.shape[1]))
   gkf = ft(gk, (blurred.shape[0],blurred.shape[1]))
   imconvf = imf / gkf
   recovered = ift(imconvf)
   
   img_out = recovered
   img_out = np.array(img_out*255).astype('uint8')
   img_out = np.clip(img_out, 0, 255)
   return True, img_out

def Question2():

   # Read in input images
   # input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   # input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   # output_name1 = sys.argv[4] + "2.jpg"
   # output_name2 = sys.argv[4] + "3.jpg"
   # output_name3 = sys.argv[4] + "4.jpg"
   output_name1 = sys.argv[4] + "output2LPF.png"
   output_name2 = sys.argv[4] + "output2HPF.png"
   output_name3 = sys.argv[4] + "output2deconv.png"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):
   if img_in1.shape[0] <= img_in1.shape[1]:
       a_min = img_in1.shape[0]
       a_index = 0
   else:
       a_min = img_in1.shape[1]
       a_index = 1

   if img_in2.shape[0] <= img_in2.shape[1]:
       b_min = img_in2.shape[0]
       b_index = 0
   else:
       b_min = img_in2.shape[1]
       b_index = 1

   if a_min <= b_min:
       if a_index == 0:
           img_in1 = img_in1[:,:img_in1.shape[0]]
           img_in2 = img_in2[:img_in1.shape[0],:img_in1.shape[0]]
       else:
           img_in1 = img_in1[:img_in1.shape[1],:]
           img_in2 = img_in2[:img_in1.shape[1],:img_in1.shape[1]]
   elif a_min > b_min:
       if b_index == 0:
           img_in2 = img_in2[:,:img_in2.shape[0]]
           img_in1 = img_in2[:img_in2.shape[0],:img_in2.shape[0]]
       else:
           img_in2 = img_in2[:img_in2.shape[1],:]
           img_in1 = img_in2[:img_in2.shape[1],:img_in2.shape[1]]
   else:
       print "some problem"

   G = img_in1.copy()
   gpA = [G]
   for i in xrange(6):
       G = cv2.pyrDown(G)
       gpA.append(G)
 
   G = img_in2.copy()
   gpB = [G]
   for i in xrange(6):
       G = cv2.pyrDown(G)
       gpB.append(G)
 
   lpA = [gpA[5]]
   for i in xrange(5,0,-1):
       GE = cv2.pyrUp(gpA[i])
       L = cv2.subtract(gpA[i-1],GE)
       lpA.append(L)
 
   lpB = [gpB[5]]
   for i in xrange(5,0,-1):
       GE = cv2.pyrUp(gpB[i])
       L = cv2.subtract(gpB[i-1],GE)
       lpB.append(L)
 
   LS = []
   for la,lb in zip(lpA,lpB):
       rows,cols,dpt = la.shape
       ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
       LS.append(ls)
 
   ls_ = LS[0]
   for i in xrange(1,6):
       ls_ = cv2.pyrUp(ls_)
       ls_ = cv2.add(ls_, LS[i])

   img_out = ls_
   return True, img_out

def Question3():

   # Read in input images
   # input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   # input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   # output_name = sys.argv[4] + "5.jpg"
   output_name = sys.argv[4] + "output3.png"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
 