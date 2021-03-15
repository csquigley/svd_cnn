import numpy as np

class SVDConverter:
  def __init__(self,dataset,pk,print_please=False):
    self.dataset = dataset
    self.pk = pk
    self.print_please = print_please
  #new_x_train is the dataset of lowrank images created from the original cifar-10 images
  def convert(self):
    new_x_train = np.empty(self.dataset.shape)
    for top_num in range(0,len(self.dataset)):
        if self.print_please == True:
          if top_num % 1000 == 0:
              print("processed: {} of {}".format(str(top_num),str(len(self.dataset))))
        current_img = self.dataset[top_num]
        current_img = current_img.transpose()
        low_rank_img = np.empty((0,32,32))

        for i_n, layer in enumerate(current_img):

            U,s,Vt = np.linalg.svd(layer)

            #this finds the total of all the singular values, which is needed to find our percentage
            total_s = 0
            for value in s:
                total_s += value
            #current percentage
            cp = 0
            #32 is just the original image
            #anything less is a low rank approximation of the image.
            i = 1
            #find the index of the singular value that will give us the desired level of information.
            # while cp < self.pk:
            #     num = s[i]
            #     cp += (num / total_s)
            #     i += 1
            #create an array of matrices that contains the desired images
            #first create a numpy array of zeros that contains i (the number of significant singular values)
            mats = np.zeros((i,32,32))
            #reconstruct the layers from Matrix U, the singular values (s) and the Matrix V transpose
            for n in range(0,i):
                mats[n,:,:] = np.outer(U[:,n]*s[n],Vt[n,:])
            #reconstruct our image from the vectors and singular values that contain the desired information level
            reconstructed_layer = np.zeros((1,32,32))
            for n in range(0,i):
                reconstructed_layer += mats[n]

            low_rank_img = np.concatenate((low_rank_img,reconstructed_layer),axis=0)
        #the transpose gets the image back to the correct shape for an RGB image
        low_rank_img = low_rank_img.transpose()
        #add an additional dimension in order to concatenate to
        # low_rank_img = low_rank_img.reshape((1,32,32,3))
        new_x_train[top_num] = low_rank_img
        if top_num >= 50000:
          break
    return new_x_train
