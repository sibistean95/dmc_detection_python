from pylibdmtx.pylibdmtx import decode
import cv2 as cv
import matplotlib.pyplot as plt

def dmc_reader(image):
    img = cv.imread(image)
    decode_dmc = decode(img)

    if not decode_dmc:
        print("Data matrix code not detected!")
    else:
        for dmc in decode_dmc:
            (x, y, w, h) = dmc.rect
            cv.rectangle(img, (x-10, y-10), (x+(w+10), y+(h+10)), (255, 0, 0), 2)

            if dmc.data != "":
                print(f"Data Matrix Code: {dmc.data.decode('utf-8')}")
                break
        im = img[:,:,::-1]
        plt.imshow(im)
        plt.show()
    result = dmc.data.decode('utf-8')
    return result

if __name__ == "__main__":
    dmc_reader("img.png")