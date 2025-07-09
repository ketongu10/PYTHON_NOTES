import cv2
vidos = "/vol2/LEBEDKA/VIDEOS/PARSER/CHOSEN/polset/nng_tkrs_noyabrsk_srv01_camera_in_interesting_position-debug_case_24-10-26_12-42-53_12-43-23_null_cam14.mp4"
kuda = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tkrs_lebedka/cut"
cap = cv2.VideoCapture(vidos)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, frame0 = cap.read()
i=0
while ret:
    i+=1
    ret, frame = cap.read()
    print(i, frame)
    cv2.imwrite(f"{kuda}/{i}.jpg", frame)
