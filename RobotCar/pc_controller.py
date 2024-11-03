import zmq
import cv2
import numpy as np
import threading
import time
import socket


IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)
ORIGINAL_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)

running = True
THRESHOLD = 128
THRESHOLD1 = 255
THRESHOLD2 = 50

# グローバル変数でキーボード入力を保存
motor_l = 0
motor_r = 0

IMAGE_CONTER = 0    # Counting the number of images
def save_image_every_30_frames():
    global IMAGE_CONTER
    if IMAGE_CONTER % 40 == 0:
        cv2.imwrite(f"./images/image_{IMAGE_CONTER}.jpg", IMAGE)
        cv2.imwrite(f"./ORIGINAL_IMAGES/image_{IMAGE_CONTER}.jpg", ORIGINAL_IMAGE)
    IMAGE_CONTER += 1

def receiver_thread():
    global IMAGE, ORIGINAL_IMAGE, running, motor_l, motor_r

    # Open ZMQ Connection
    port = 5555
    conn_str = f"tcp://*:{port}"  # Connection String
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(conn_str)
    print("Receiver start.")

    # FSP計測用の変数
    count = 0   # 画像数
    start_time = time.time()

    while running:
        # Receve Data
        try:      
            jpeg_bytes = sock.recv(flags=zmq.NOBLOCK)

            # Calculate FPS
            count += 1
            if count == 30:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = count / elapsed_time
                print(f"FSP : {fps}")

                # Reset counter and timer
                count = 0
                start_time = time.time()
            
            # 受信した画像データに対する応答としてキーボード入力の値を送信
            sock.send_string(f"{motor_l},{motor_r}")

            # バイト列からnumpy配列にデコードして画像を取得
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            gray_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            ORIGINAL_IMAGE = gray_image

            # # 二値化
            # _, binary_image = cv2.threshold(gray_image, THRESHOLD, 255, cv2.THRESH_BINARY)    # 128:閾値, 255:最大値, cv2.THRESH_BINARY:二値化

            # # edge detection
            # edges = cv2.Canny(binary_image, THRESHOLD1, THRESHOLD2) # 50:閾値1, 150:閾値2

            # # contor detection
            # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # largest_contour = max(contours, key=cv2.contourArea)

            # # 空の画像を作成（元の画像と同じサイズ、黒背景）
            # image_to_show = np.zeros_like(gray_image)

            # # 最大輪郭だけを描画（青色で）
            # cv2.drawContours(image_to_show, [largest_contour], -1, (255, 0, 0), 2)
            IMAGE = gray_image

            # # Save the image
            # save_image_every_30_frames()

            # # largest contour detection
            # # 検出されたすべての輪郭の中から最大の輪郭を取得
            # largest_contour = max(contours, key=cv2.contourArea)

        except zmq.ZMQError:
            time.sleep(0.01)
            continue

def gui():
    # Initialize
    global running, motor_l, motor_r
    window_name = "receiver"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("Press [ESC] to quit.")

    motor_l = 0
    motor_r = 0

    # GUI loop
    while running:
        cv2.imshow(window_name, IMAGE)
        key = cv2.waitKey(30)

        # モーター制御ロジックをここに統合
        if key < 0:
            motor_l = 0
            motor_r = 0
        if key == ord("w"):
            motor_l = 50
            motor_r = 47
        if key == ord("a"):
            motor_l = 0
            motor_r = 50
        if key == ord("d"):
            motor_l = 50
            motor_r = 0
        if key == ord("s"):
            motor_l = 30
            motor_r = 30
        if key == ord("e"):
            motor_l = 40
            motor_r = 30
        if key == ord("q"):
            motor_l = 30
            motor_r = 40
        if key == 27:
            running = False  # GUIを終了するための条件もここで設定

    # Closing
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    ip = socket.gethostbyname(socket.gethostname())
    print(ip)

    receiver = threading.Thread(target=receiver_thread)

    receiver.start()

    gui()

    receiver.join()