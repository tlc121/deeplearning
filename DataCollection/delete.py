import os

if __name__ == '__main__':
    path = '/Users/xavier0121/Desktop/work/patientinfo2/'
    for folder in os.listdir(path):
        if os.path.isdir(path + folder + '/'):
            for file in os.listdir(path + folder + '/'):
                os.remove(path + folder + '/' + file)
            os.rmdir(path + folder + '/')