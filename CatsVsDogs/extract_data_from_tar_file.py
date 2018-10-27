import tarfile

cat_path = './train_cat.tar.gz'
dog_path = './train_dog.tar.gz'

def extractFileToTrainAndValidation(path):
    name = path.split('.')[1][-3:]
    mone = 0
    tf = tarfile.open(path)
    for member in tf.getmembers():
        if(mone<100):
            tf.extract(member = member, path='./data/source/validation/'+ name)
            mone+=1
        else:
            tf.extract(member = member, path='./data/source/traning/' + name)