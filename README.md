# "Image algorithms for low-level vision tasks" Codebase

Source code for book *["Image algorithms for low-level vision tasks" (Jia. 2024, Publishing House of Electronics Industry, Broadview)](https://book.douban.com/subject/36895899/)*


<p align="center">
  <img src="https://img3.doubanio.com/view/subject/l/public/s34877637.jpg" height=300>

## How to use

  This repo is the implementations of the methods and algorithms introduced in the above mentioned book. If you find any errors or mistakes in the codes, or typos and tech errors in the book, feel free to open an issue to let me know. An **Erratum** will be maintained in this repo. Thanks for buying and reading this work, hope it be helpful for your studying or research!

## The structure of this codebase:

- chapter 1 (None)

- chapter 2 **Basics**

  including image transforms, histograms, color, and frequency analysis
  ![histogram](ch2_basics/results/hist/hist_compare.png)

- chapter 3 **Denoise**

  including classical denoising methods (Gaussian/wavelet/BM3D etc.) and DL based denoising method (DnCNN/FFDNet etc.)
  ![guided_filter](ch3_denoise/results/guided_filter/guided.png)

- chapter 4 **Super-Resolution**

  including classical enhancing and DL based SR methods and network implementations (upsampling/USM and SRCNN/RCAN/EDSR etc.)
  ![USM](ch4_super/results/usm/usm_result.png)

- chapter 5 **Dehazing**

  including dehazing methods and networks (dark channel prior, DehazeNet etc.)

- chapter 6 **HDR**

  including classical HDR methods and DL based networks related to HDR tasks
  ![reinhard](ch6_hdr/results/reinhard/reinhard_out.png)

- chapter 7 **Composition**

  including alpha blending, laplacian blending and poisson blending, and image harmonization networks
  ![image ref: [link](https://github.com/willemmanuel/poisson-image-editing/tree/master/input/2)](ch7_composite/results/copy_paste.png) 

- chapter 8 **Enhancement**

  including low-light enhancement and color enhancement, retouch methods
  ![lowlight enhancement](ch8_enhance/results/invert_dehaze/out.png)



## Update Logs

1. [2024-03-30] initial upload.
2. [2024-06-06] book information added.

