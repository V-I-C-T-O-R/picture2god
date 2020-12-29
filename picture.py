import numpy as np
import cv2 as cv


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def canny():
    img = cv.imread('test1.png', cv.IMREAD_GRAYSCALE)
    canny_img = cv.Canny(img, 200, 150)
    cv.imwrite('test2.png', canny_img)

def mark():
    img = cv.imread('test2.png', 1)
    k = np.ones((3, 3), np.uint8)
    img2 = cv.morphologyEx(img, cv.MORPH_CLOSE, k)  # 闭运算
    cv.imwrite('test3.png', img2)

def SSR(src_img, size):
    L_blur = cv.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv.log(img / 255.0)
    dst_Lblur = cv.log(L_blur / 255.0)
    dst_IxL = cv.multiply(dst_Img, dst_Lblur)
    log_R = cv.subtract(dst_Img, dst_IxL)

    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8

def repair(path):
    img = cv.imread(path)
    b = cv.imread('test3.png',0)
    dst = cv.inpaint(img, b, 5, cv.INPAINT_TELEA)
    cv.imshow('dst', dst)
    cv.imwrite(f'repair_{path}', dst)
    cv.waitKey()
    cv.destroyAllWindows()

def enhance():
    img = 'test.png'
    size = 3
    src_img = cv.imread(img)
    b_gray, g_gray, r_gray = cv.split(src_img)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv.merge([b_gray, g_gray, r_gray])

    cv.imshow('test.png', src_img)
    cv.imshow('test1.png', result)
    cv.imwrite('test1.png', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    enhance()
    canny()
    mark()
    repair('test.png')