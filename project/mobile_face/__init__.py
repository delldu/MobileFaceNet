"""Mobile Face Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

############################ mobile_face API ############################
#
# d = mobile_face.dector()
# hasface, bboxes, landms = d(input_tensor)

# e = mobile_face.extractor()
# f1 = e(face_tensor1)
# f2 = e(face_tensor1)
# is_same_face = e.verify(f1, f2)
#
# Swap face
# face1, landms1 = mobile_face.get(input_tensor1, bbox1, landmark1)
# face2, landms2 = mobile_face.get(input_tensor2, bbox2, landmark2)
# new_face = mobile_face.transform(face1, landms1, face2, landms2)
# mobile_face.set(input_tensor, bbox1, new_face)
# 
##########################################################################


from torchvision import transforms as T

DEFAULT_FACE_SIZE = (112, 112)

def face_tensor(image):
    '''image is Pillow image object, return 1xCxHxW tensor'''

    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return t(image)

    # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # img -= (104, 117, 123)
    # img = img.transpose(2, 0, 1)
    # img = torch.from_numpy(img).unsqueeze(0)

def detector(device=torch.device("cuda")):

    return None

def extractor(device=torch.device("cuda")):
    return None

def get(image_tensor, bbox, landmark):
    return None

def set(image_tensor, bbox, new_sub):
    pass

def transform(sface_tensor, slandmark, dface_tensor, dlandmark):
    '''New face tensor, size like as dst_face_tensor'''
    return None

