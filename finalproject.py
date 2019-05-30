#-*- coding: utf-8 -*-
#파일에 한글을 인식하기 위해 사용한코드
'''
프로젝트 실행 전 준비되어야 할 것들

각종 관련 라이브러리와
leptonica 설치 , tesseract 설치하고
tesseract 버젼과 일치하는 한글인식파일을 설치합니다.
그 파일을 /usr/local/share/tesseract/에 압축을 풀어줍니다
export TESSDATA_PREFIX=/usr/local/share/
라고 환경변수를 등록해줍니다

그리고 라즈베리파이가 아닌 파이썬 파일에서 사용하기위해

tesseract와 mapping되는 pytesseract 라이브러리 설치를합니다 .

'''
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import requests
import pytesseract
from picamera import PiCamera
from time import sleep
from PIL import Image
from StringIO import StringIO
import cv2

 

def order_points(pts):

    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
def auto_scan_image():
    #처음에 라즈베리 카메라 사용 하기 위해 사용한코드
    #라즈베리파이 카메라 고장으로 주석처리
    #camera=PiCamera()
    #camera.start_preview()
    #sleep(3)
    #camera.capture('/home/pi/tesseract-3.04.01/imagee.jpg')
    #camera.stop_preview()
    image = cv2.imread('project4.jpg')
    #projcet1.jpg project2.png project3.png project4.jpg  project5.jpg 사진 종류가 있습니다.
    '''project1 은 글자를 먼저 인식하기 위해 사용한 이미지파일이고
       project2는 색깔있는 배경의 글자를 인식하기 위해 사용했습니다.
       project3은 색깔 여러개를 주어 글자를 인식하였고
       project4는 같은 문자열을 2번씩 실험해봤습니다.
       project5는 실제 아이스크림 봉지를 인식해봤는데 여기까진 미구현 상태입니다.'''
    orig = image.copy()
    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("edge", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    rect = order_points(screenCnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    dst = np.float32([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    #cv2.imshow("Warped.png", warped)
    #cv2.imwrite('Warped.png',warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    warped = cv2. cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    cv2.imshow("original",orig)
    cv2.imshow("ScanImg", warped)
    cv2.imwrite('scan.png', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #pytesseract 사용해서 한글 추출 코드
    result = pytesseract.image_to_string('scan.png', lang='kor')
    #print(result.replace(" ",""))

    #아이스크림 리스트 [name]=price
    icecream={}
    icecream['메로나']='700'
    icecream['더위사냥']='1500'
    icecream['누가바']='700'
    icecream['비비빅']='700'
    icecream['와일드바디']='900'
    icecream['옥동자']='800'
    icecream['요맘때']='700'
    icecream['붕어싸만코']='1300'
    result=' '.join(result.split())
    hap=0
    print(result)
    for s in icecream:
        if s in result:
            c=result.count(s)
            print(s.ljust(10,' ') +"(수량 "+str(c)+"개) : "+str(c*int(icecream[s]))+"원")
            won=c*int(icecream[s])
            hap+=won
    print("total cost".ljust(10)+" : "+str(hap)+"원")
    
    
if __name__ == '__main__':
    auto_scan_image()
