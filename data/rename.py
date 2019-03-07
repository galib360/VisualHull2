
# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
    cam = 0
    for cam in range(8):
        i = 0
        for filename in os.listdir("cam0"+str(cam)): 
            dst = str(i) + ".pbm"
            src ='cam0'+str(cam)+"/"+ filename 
            dst ='cam0'+str(cam)+"/"+ dst 
          
            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 
            i += 1
  
    
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 

