## README

Mobile face 1.0.0 package

### 1. How to use ?
demo.py is answer.

### 2. Reference

```
from PIL import Image
import mobile_face

imge = Image.open(filename).convert("RGB")
input_tensor = mobile_face.tensor(image)

d = mobile_face.dector()
hasface, dets, landms = d(input_tensor)
mobile_face.draw(image, dets, landms)

faces = mobile_face.align(image, landms)

face_tensor1 = mobile_face.tensor(faces[0])
face_tensor2 = mobile_face.tensor(faces[1])

e = mobile_face.extractor()
f1 = e(face_tensor1)
f2 = e(face_tensor2)
is_same_face = e.verify(f1, f2)
```