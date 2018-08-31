import cv2
import numpy as np
import os

class globalvalue:
    eigenface_mat_gv = []
    component_set_gv = []
    k_gv = 0
    mean_gv = []
    sample_num_gv = 0
    map_gv = {}

def prepeocess(image):
    img_crop = image[90:500, 450:850]
    img_rescale = cv2.resize(img_crop, (92, 112), interpolation=cv2.INTER_CUBIC)
    return img_rescale


def loadnewfaces(filepath, filepath1, testpath):
    #get the number of files in the directory
    filenum = len(os.listdir(filepath1)) - 2

    #create new files to store the new faces
    newfiledir = filepath1 + 's' + str(filenum+1) + '/'
    newfiledir_test = testpath + 's' + str(filenum+1) + '/'
    os.makedirs(newfiledir)
    os.makedirs(newfiledir_test)

    #put the new faces in files
    count = 1
    for filename in os.listdir(filepath):
        for img in range(0, 7):
            if filename.endswith(str(img)+'.pgm'):
                image = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
                img_preprocess = prepeocess(image)
                cv2.imwrite(newfiledir + str(count) + '.pgm', img_preprocess)
                count += 1
        for testimg in range(7, 10):
            if filename.endswith(str(testimg) + '.pgm'):
                image = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
                img_preprocess = prepeocess(image)
                cv2.imwrite(newfiledir_test + str(count) + '.pgm', img_preprocess)
                count += 1



def calculate_k(eigenval, dimension):
    summation = sum(eigenval)
    threshold = 1
    temp = 0
    count = 0
    for i in range(dimension - 1, 0, -1):
        if temp/summation < threshold:
            temp = temp + eigenval[i]
            count = count + 1
        else:
            break
    return count

def img2vector(img):
    row, col = img.shape
    result = np.zeros(shape=[row*col, 1])
    for i in range(0, row - 1):
        for j in range(0, col - 1):
            result[i*col + j, 0] = img[i, j]
    return result

def vector2image(vec):
    row = 112
    col = 92
    res = np.zeros(shape=[row, col])
    for i in range (0,row - 1):
        for j in range(0, col - 1):
            res[i, j] = vec[i*col + j]
    return res


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float32') - min_val) / (max_val - min_val)
    return out


def averageface(filepath1, Allimage, map):
    temp = np.zeros(shape=[112, 92])
    #get the filenum in the filepath
    filenum = len(os.listdir(filepath1)) - 2
    count = 0
    for num in range (1, filenum + 1):
        for filename in os.listdir(filepath1 + 's' + str(num) + '/'):
            if filename.endswith('.pgm'):
                map.update({count: str(num)+'.pgm'})
                count += 1
                image = cv2.imread(filepath1 + 's' + str(num) + '/' + filename, cv2.IMREAD_GRAYSCALE)
                row, col = image.shape
                im_double = im2double(image)
                Allimage.append(img2vector(im_double))
                temp = temp + im_double
    mean = img2vector( temp / float(count))
    return mean, row, col, count


def calculateEigen(covariance):
    eigenvalue, eigenvector = np.linalg.eig(covariance);
    return eigenvalue, eigenvector


def calculate_the_component(filepath1):
    Allimage = []
    map = {}

    #calculate the mean face
    mean, row, col, count = averageface(filepath1, Allimage, map)

    #construct the matrix A and At
    A_mat = np.zeros(shape = [row*col, count])
    original_image = np.zeros(shape = [row*col, count])
    counter = 0
    for image in Allimage:
        A_mat[:, counter:counter+1] = image - mean
        original_image[:, counter:counter+1] = image
        counter += 1
    At_mat = np.transpose(A_mat)


    #compute the eigenvalue and eigenvector
    covariance = np.dot(At_mat, A_mat)
    eigenvalue, eigenvector = calculateEigen(covariance)

    #calculate the K
    sorted_eigenval = np.sort(eigenvalue)
    dimens = eigenvalue.shape[0]
    k = calculate_k(sorted_eigenval, dimens)

    # sort the indices of eigenvalue
    sorted_indices = np.argsort(eigenvalue)
    topk_eigenv = eigenvector[:, sorted_indices[:-k-1:-1]]
    eigenface_mat = np.dot(A_mat, topk_eigenv)

    #store the wi for each face
    component_set = np.zeros(shape = [k, count])
    for number in range(0, count):
        component = np.zeros(shape = [k, 1])
        summ = 0
        for i in range(0, k):
            wi = np.dot(np.transpose(eigenface_mat[:, i:i+1]), A_mat[:, number:number+1])
            summ += wi

        for i in range(0, k):
            wi = np.dot(np.transpose(eigenface_mat[:, i:i+1]), A_mat[:, number:number+1])
            component[i, 0] = float(wi)/float(summ)
        component_set[:, number:number+1] = component

    return component_set, mean, k, eigenface_mat, count, original_image, map


def reconstruct(eigenface_mat, component, k, mean):
    recon = np.zeros(shape = [112*92, 1])
    for i in range(0, k):
        recon = recon + component[i, 0] * eigenface_mat[:, i:i+1]
    return vector2image(recon)


def mahanalobis(v1, v2, cov):
    temp = np.dot(np.transpose(v1 - v2), np.linalg.inv(cov))
    distance = np.dot(temp, (v1 - v2))
    return np.abs(distance)

def Recognition(image, component_set, k, eigenface_mat, sample_num):

    #calculate the component of new image
    summ = 0
    component = np.zeros(shape = [k, 1])
    for i in range(0, k):
        wi = np.dot(np.transpose(eigenface_mat[:, i:i + 1]), image)
        summ += wi
    for i in range(0, k):
        wi = np.dot(np.transpose(eigenface_mat[:, i:i + 1]), image)
        component[i, 0] = wi / summ

    #compare with whole dataset
    min = 1000000
    target = 0
    for i in range(0, sample_num):
        comp_temp = component_set[:, i:i+1]
        norm2 = np.linalg.norm(component - comp_temp, ord = 2)
        diff = np.sqrt(norm2)
        if diff < min:
            min = diff
            target = i
    return target


def accuracy(filepath, map, mean, sample_num, k, component_set, eigenface_mat):
    wholesample = 0
    correct = 0
    for i in range(1, 41):
        filepath1 = filepath + 's' + str(i) + '/'
        for filename in os.listdir(filepath1):
            if filename.endswith('.pgm'):
                wholesample += 1
                image = cv2.imread(filepath1 + filename, cv2.IMREAD_GRAYSCALE)
                im_double = im2double(image)
                img_v = img2vector(im_double)
                result = Recognition(img_v - mean, component_set, k, eigenface_mat, sample_num)
                output = map[result]
                if output == str(i) + '.pgm':
                    correct += 1
    acc = float(correct)/float(wholesample)
    return acc


def traningdata():
    filepath1 = '/Users/xavier0121/Desktop/training/orl_faces/'
    component_set, mean, k, eigenface_mat, sample_num, original_face, map = calculate_the_component(filepath1)
    globalvalue.component_set_gv = component_set
    globalvalue.eigenface_mat_gv = eigenface_mat
    globalvalue.sample_num_gv = sample_num
    globalvalue.map_gv = map
    globalvalue.k_gv = k
    globalvalue.mean_gv = mean


def loadingfaces(filepath):
    testpath = '/Users/xavier0121/Desktop/test_recog/orl_faces/'
    filepath1 = '/Users/xavier0121/Desktop/training/orl_faces/'
    loadnewfaces(filepath, filepath1, testpath)











