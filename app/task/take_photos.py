import cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
flag = 1 
num = 1
while(cap.isOpened()):
    ret_flag, Vshow = cap.read()
    cv2.imshow("Capture_Test",Vshow)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'): 
        cv2.imwrite("yolo"+ str(num) + ".jpg", Vshow)
        print(cap.get(3)); 
        print(cap.get(4));
        print("success to save"+str(num)+".jpg")
        print("-------------------------")
        num += 1
    elif k == ord('q'): 
        break
cap.release() 
cv2.destroyAllWindows()

