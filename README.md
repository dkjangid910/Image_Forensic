# Copy Paste Forgery detection using EM Algorithm
The task of this Project was to detect Copy-Paste forgery without using Deep Learning. I have used Expectation-Maximization algorithm given by [A.C. Popescu](https://ieeexplore.ieee.org/document/1381775) to detect forgery.  A probability map is generated as an output of EM algorithm. Fourier transform of probability map is calculated for small patches of the Image, which include both modified part and original part of the image, to detect forgery.

## Required Libraries:
   1. OpenCv 3.4.2.16 (you can install this version using pip3) 
   2. python 3.6 
  
## How to Run Code:
   1. Clone Repo
   ```
   git clone "https://github.com/dkjangid910/Image_Forensic.git"
   ```
   2. Create Virtual environment
   ```
    virtualenv -p /usr/bin/python3.6 venv(name of virtual environment)
   ```
   3. Activate Virtual environment
   ```
   source venv/bin/activate
   ```
   4. Download dependencies 
   ```
   pip install -r requirements.txt 
   ```
   5. Run Code
   ```
   python EM_algo.py -i ./Data/1.jpg 
   ``` 
   Probability Map image will be saved as 1_prob_map.jpg and fourier transform results will be saved in result folder.
