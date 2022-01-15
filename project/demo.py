"""Mobile Face Demo."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#
import os
import argparse
from PIL import Image

import mobile_face

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default="images/lena.png", help="input file")
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")

    detector = mobile_face.detector()
    input_tensor = mobile_face.tensor(image)

    input_tensor = input_tensor.to(detector.device)

    hasface, dets, landms = detector(input_tensor)

    # print(detector.backbone)
    print("input_tensor: ", input_tensor.size())
    print("detect: ", hasface)
    print("bboxes: ", dets)
    print("landms: ", landms)

    if hasface:
        mobile_face.draw(image, dets, landms)
    image.show()

    if hasface:
        extractor = mobile_face.extractor()
        # print(extractor.backbone)

        faces = mobile_face.align(image, landms)
        for fimage in faces:
            fimage.show()
            ftensor = mobile_face.tensor(fimage).to(extractor.device)
            f = extractor(ftensor)
            print("face feature size: ", f.size())
