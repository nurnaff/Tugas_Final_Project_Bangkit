from django.shortcuts import render

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image 
import tensorflow as tf
from tensorflow import Graph,Session
import json

img_height,img_width=224,224
with open('./models/model_ku.json','r') as f:
    labelInfo=f.read()
labelInfo=json.loads(labelInfo)

model_graph=Graph()
with model_graph.as_default():
    tf_session=Session()
    with tf_session.as_default():
        model=load_model('./models/model.h5')



# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName

    img=image.load_img(testimage,target_size=(img_height,img_width))
    x=image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height,img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

    import numpy as np
    #predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    #print(predi)
    if(predi<0.8):
        predictedLabel="Benign"
    else:
        predictedLabel="Malignant"

    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    #context={'filePathName':filePathName,'predi':predi}
    return render(request,'index.html',context)
